from __future__ import annotations

import torch
from torch import nn

from vgt_inference.core.structures import Boxes, Instances
from vgt_inference.modeling.box_regression import Box2BoxTransform
from vgt_inference.modeling.nms import batched_nms


class StandardRPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, conv_dims: int = -1):
        super().__init__()
        if conv_dims < 0:
            conv_dims = in_channels
        self.conv = nn.Conv2d(in_channels, conv_dims, kernel_size=3, padding=1)
        self.objectness_logits = nn.Conv2d(conv_dims, num_anchors, kernel_size=1, stride=1)
        self.anchor_deltas = nn.Conv2d(conv_dims, num_anchors * 4, kernel_size=1, stride=1)

        for layer in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: list[torch.Tensor]):
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = torch.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


class RPN(nn.Module):
    def __init__(
        self,
        *,
        in_features: list[str],
        head,
        anchor_generator,
        anchor_matcher,
        box2box_transform,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        pre_nms_topk: tuple[int, int] = (12000, 6000),
        post_nms_topk: tuple[int, int] = (2000, 1000),
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        boundary_threshold: int = 0,
        loss_weight: float = 1.0,
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size
        self.boundary_threshold = boundary_threshold
        self.loss_weight = loss_weight
        self.box_reg_loss_type = box_reg_loss_type
        self.smooth_l1_beta = smooth_l1_beta

    def forward(self, images, features, gt_instances=None):
        features_list = [features[f] for f in self.in_features]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features_list)
        anchors = self.anchor_generator(features_list)
        
        outputs = find_top_rpn_proposals(
            anchors,
            pred_objectness_logits,
            pred_anchor_deltas,
            images.image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[1],
            self.post_nms_topk[1],
            self.min_box_size,
            self.box2box_transform,
        )
        
        if self.training:
            return outputs, {}
        return outputs, {}


def find_top_rpn_proposals(
    anchors,
    pred_objectness_logits,
    pred_anchor_deltas,
    image_sizes,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_size,
    box2box_transform,
):
    """
    Re-written to match detectron2's find_top_rpn_proposals exactly:
    1.  Apply deltas per level to get proposal boxes.
    2.  Select top-k proposals *per level* before NMS.
    3.  Concatenate all levels and run per-level NMS (using level_ids).
    4.  Keep top-k proposals *total* after NMS.
    """
    num_images = len(image_sizes)
    device = pred_objectness_logits[0].device

    # Decode deltas into proposal boxes per level
    # We must reshape deltas exactly as detectron2 does so that the flattened
    # ordering is (H, W, A) rather than (A, H, W).
    proposals_per_level = []
    for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
        # anchors_i: list[Tensor] of length N, each (Hi*Wi*A, 4)
        # pred_anchor_deltas_i: (N, A*4, Hi, Wi)
        N = pred_anchor_deltas_i.shape[0]
        B = anchors_i[0].shape[1]  # box dim (4)
        # detectron2: view(N, A, B, H, W) -> permute(0, 3, 4, 1, 2) -> flatten(1, -2)
        pred_anchor_deltas_i = (
            pred_anchor_deltas_i.view(N, -1, B, pred_anchor_deltas_i.shape[-2], pred_anchor_deltas_i.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
        )  # (N, Hi*Wi*A, 4)
        # Expand anchors for batch
        anchors_expanded = torch.stack(anchors_i, dim=0)  # (N, Hi*Wi*A, 4)
        proposals_i = box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_expanded)
        proposals_per_level.append(proposals_i)  # (N, Hi*Wi*A, 4)

    # Reshape logits to (N, Hi*Wi*A)
    logits_per_level = []
    for logits in pred_objectness_logits:
        # logits: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        logits_per_level.append(logits.permute(0, 2, 3, 1).flatten(1))

    # 1. Select top-k per level and per image
    topk_scores = []
    topk_proposals = []
    level_ids = []
    batch_idx = torch.arange(num_images, device=device)
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals_per_level, logits_per_level)):
        num_proposals_i = min(logits_i.shape[1], pre_nms_topk)
        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # (N, topk, 4)
        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device)
        )

    # 2. Concatenate all levels
    topk_scores = torch.cat(topk_scores, dim=1)      # (N, total_topk)
    topk_proposals = torch.cat(topk_proposals, dim=1)  # (N, total_topk, 4)
    level_ids = torch.cat(level_ids, dim=0)            # (total_topk,)

    # 3. For each image, clip, filter, and NMS
    proposals = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]

        boxes.clip(image_size)

        keep = boxes.nonempty(threshold=min_box_size)
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], lvl[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        keep = keep[:post_nms_topk]

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        proposals.append(res)

    return proposals
