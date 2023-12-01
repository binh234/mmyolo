# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmdet.models.backbones.csp_darknet import Focus
from mmdet.models.layers import ChannelAttention
from mmengine.config import ConfigDict
from torch import Tensor

from mmyolo.models import RepVGGBlock
from mmyolo.models.dense_heads import YOLOXPoseHead
from mmyolo.models.layers import ImplicitA, ImplicitM
from ..backbone import DeployFocus, GConvFocus, NcnnFocus
from ..nms import batched_nms, efficient_nms, onnx_nms
from .backend import MMYOLOBackend


class DeployPoseModel(nn.Module):
    transpose = False

    def __init__(
        self,
        baseModel: nn.Module,
        backend: MMYOLOBackend,
        postprocess_cfg: Optional[ConfigDict] = None,
    ):
        super().__init__()
        self.baseModel = baseModel
        self.baseHead = baseModel.bbox_head
        self.backend = backend
        if postprocess_cfg is None:
            self.with_postprocess = False
        else:
            self.with_postprocess = True
            self.__init_sub_attributes()
            self.detector_type = type(self.baseHead)
            self.pre_top_k = postprocess_cfg.get("pre_top_k", 1000)
            self.keep_top_k = postprocess_cfg.get("keep_top_k", 100)
            self.iou_threshold = postprocess_cfg.get("iou_threshold", 0.65)
            self.score_threshold = postprocess_cfg.get("score_threshold", 0.25)
        self.__switch_deploy()

    def __init_sub_attributes(self):
        self.bbox_decoder = self.baseHead.bbox_coder.decode
        self.pose_decoder = self.baseHead.decode_pose
        self.prior_generate = self.baseHead.prior_generator.grid_priors
        self.num_base_priors = self.baseHead.num_base_priors
        self.featmap_strides = self.baseHead.featmap_strides
        self.num_classes = self.baseHead.num_classes
        self.num_keypoints = self.baseHead.num_keypoints

    def __switch_deploy(self):
        headType = type(self.baseHead)

        if self.backend in (MMYOLOBackend.HORIZONX3, MMYOLOBackend.NCNN, MMYOLOBackend.TORCHSCRIPT):
            self.transpose = True
        for layer in self.baseModel.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, ChannelAttention):
                layer.global_avgpool.forward = self.forward_gvp
            elif isinstance(layer, Focus):
                # onnxruntime openvino tensorrt8 tensorrt7
                if self.backend in (
                    MMYOLOBackend.ONNXRUNTIME,
                    MMYOLOBackend.OPENVINO,
                    MMYOLOBackend.TENSORRT8,
                    MMYOLOBackend.TENSORRT7,
                ):
                    self.baseModel.backbone.stem = DeployFocus(layer)
                # ncnn
                elif self.backend == MMYOLOBackend.NCNN:
                    self.baseModel.backbone.stem = NcnnFocus(layer)
                # switch focus to group conv
                else:
                    self.baseModel.backbone.stem = GConvFocus(layer)

    def pred_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        objectnesses: Optional[List[Tensor]] = None,
        kpt_preds: Optional[List[Tensor]] = None,
        vis_preds: Optional[List[Tensor]] = None,
        **kwargs
    ):
        assert len(cls_scores) == len(bbox_preds)
        dtype = cls_scores[0].dtype
        device = cls_scores[0].device
        batch_size = bbox_preds[0].shape[0]
        bbox_decoder = self.bbox_decoder

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        mlvl_priors = self.prior_generate(
            featmap_sizes, dtype=dtype, device=device)

        flatten_priors = torch.cat(mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full((featmap_size.numel(),), stride)
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for bbox_pred in bbox_preds
        ]
        flatten_vis_preds = [
            vis_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_keypoints)
            for vis_pred in vis_preds
        ]
        flatten_kpt_preds = [
            kpt_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_keypoints * 2)
            for kpt_pred in kpt_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_kpt_preds = torch.cat(flatten_kpt_preds, dim=1)
        flatten_vis_preds = torch.cat(flatten_vis_preds, dim=1).sigmoid()

        bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if objectnesses is not None:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(batch_size, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
            flatten_cls_scores = flatten_cls_scores * (flatten_objectness.unsqueeze(-1))

        scores = flatten_cls_scores
        flatten_decoded_kpts = self.decode_pose(flatten_priors, flatten_kpt_preds, flatten_stride)
        pred_kpts = torch.cat([flatten_decoded_kpts, flatten_vis_preds.unsqueeze(3)], dim=3)

        bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds, flatten_stride)

        # topk
        max_scores, _ = scores.max(-1)
        _, keep_indices = torch.topk(max_scores, self.pre_top_k, dim=1)
        batch_inds = torch.arange(batch_size, device=scores.device).view(-1, 1)
        dets = torch.cat([bboxes, scores], dim=2)
        dets = dets[batch_inds, keep_indices, ...]
        pred_kpts = pred_kpts[batch_inds, keep_indices, ...]

        return dets, pred_kpts

    def decode_pose(self, grids: torch.Tensor, offsets: torch.Tensor,
                    strides: Union[torch.Tensor, int]) -> torch.Tensor:
        """Decode regression offsets to keypoints.

        Args:
            grids (torch.Tensor): The coordinates of the feature map grids.
            offsets (torch.Tensor): The predicted offset of each keypoint
                relative to its corresponding grid.
            strides (torch.Tensor | int): The stride of the feature map for
                each instance.
        Returns:
            torch.Tensor: The decoded keypoints coordinates.
        """

        if isinstance(strides, int):
            strides = torch.tensor([strides]).to(offsets)

        strides = strides.reshape(1, -1, 1, 1)
        offsets = offsets.reshape(*offsets.shape[:2], -1, 2)
        xy_coordinates = (offsets[..., :2] * strides) + grids.unsqueeze(1)
        return xy_coordinates

    def select_nms(self):
        if self.backend in (MMYOLOBackend.ONNXRUNTIME, MMYOLOBackend.OPENVINO):
            nms_func = onnx_nms
        elif self.backend == MMYOLOBackend.TENSORRT8:
            nms_func = efficient_nms
        elif self.backend == MMYOLOBackend.TENSORRT7:
            nms_func = batched_nms
        else:
            raise NotImplementedError

        return nms_func

    def forward(self, inputs: Tensor):
        neck_outputs = self.baseModel(inputs)
        if self.with_postprocess:
            return self.pred_by_feat(*neck_outputs)
        else:
            outputs = []
            if self.transpose:
                for feats in zip(*neck_outputs):
                    if self.backend in (MMYOLOBackend.NCNN, MMYOLOBackend.TORCHSCRIPT):
                        outputs.append(torch.cat([feat.permute(0, 2, 3, 1) for feat in feats], -1))
                    else:
                        outputs.append(torch.cat(feats, 1).permute(0, 2, 3, 1))
            else:
                for feats in zip(*neck_outputs):
                    outputs.append(torch.cat(feats, 1))
            return tuple(outputs)

    @staticmethod
    def forward_single(x: Tensor, convs: nn.Module) -> Tuple[Tensor]:
        if isinstance(convs, nn.Sequential) and any(
            type(m) in (ImplicitA, ImplicitM) for m in convs
        ):
            a, c, m = convs
            aw = a.implicit.clone()
            mw = m.implicit.clone()
            c = deepcopy(c)
            nw, cw, _, _ = c.weight.shape
            na, ca, _, _ = aw.shape
            nm, cm, _, _ = mw.shape
            c.bias = nn.Parameter(
                c.bias + (c.weight.reshape(nw, cw) @ aw.reshape(ca, na)).squeeze(1)
            )
            c.bias = nn.Parameter(c.bias * mw.reshape(cm))
            c.weight = nn.Parameter(c.weight * mw.transpose(0, 1))
            convs = c
        feat = convs(x)
        return (feat,)

    @staticmethod
    def forward_gvp(x: Tensor) -> Tensor:
        return torch.mean(x, [2, 3], keepdim=True)
