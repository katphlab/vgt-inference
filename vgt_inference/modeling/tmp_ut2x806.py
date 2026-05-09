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
    # anchors: list of [num_levels] tensors, each (num_anchors_all_images, 4)
    # Need to group by image first
    num_images = len(image_sizes)
    
    # Split anchors per image
    anchors_per_image = []
    for level_idx, anchors_level in enumerate(anchors):
        if level_idx == 0:
            num_anchors_per_level = anchors_level.shape[0] // num_images
        # Split into per-image anchors
        anchors_split = anchors_level.reshape(num_images, -1, 4)
        for img_idx in range(num_images):
            if level_idx == 0:
                anchors_per_image.append([])
            anchors_per_image[img_idx].append(anchors_split[img_idx])
    
    proposals = []
    for img_idx, img_size in enumerate(image_sizes):
        # Collect predictions from all feature levels for this image
        logit_list = []
        delta_list = []
        for logits, deltas in zip(pred_objectness_logits, pred_anchor_deltas):
            # logits: (N, A, Hi, Wi), deltas: (N, A*4, Hi, Wi)
            N, A, H, W = logits.shape
            logit_list.append(logits[img_idx].permute(1, 2, 0).reshape(-1))
            delta_list.append(deltas[img_idx].permute(1, 2, 0).reshape(-1, 4))
        
        objectness = torch.cat(logit_list)  # (num_anchors_total,)
        deltas = torch.cat(delta_list)  # (num_anchors_total, 4)
        
        # Convert anchors to boxes
        anchors_img = torch.cat(anchors_per_image[img_idx])
        
        # Apply deltas
        print(f'DEBUG inside: deltas={deltas.shape[0]}, anchors_img={anchors_img.shape[0]}')
    assert deltas.shape[0] == anchors_img.shape[0], f'deltas={deltas.shape[0]} != anchors={anchors_img.shape[0]}'
    proposal_boxes = box2box_transform.apply_deltas(deltas.unsqueeze(0), anchors_img.unsqueeze(0))[0]
        
        # Clip to image
        proposal_boxes[:, 0].clamp_(min=0, max=img_size[1])
        proposal_boxes[:, 2].clamp_(min=0, max=img_size[1])
        proposal_boxes[:, 1].clamp_(min=0, max=img_size[0])
        proposal_boxes[:, 3].clamp_(min=0, max=img_size[0])
        
        # Filter small boxes
        keep = Boxes(proposal_boxes).nonempty(threshold=min_box_size)
        if keep.sum().item() == 0:
            # Create empty instances
            inst = Instances(img_size)
            inst.proposal_boxes = Boxes(torch.empty(0, 4))
            inst.objectness_logits = torch.empty(0)
            proposals.append(inst)
            continue
            
        proposal_boxes = proposal_boxes[keep]
        objectness = objectness[keep]
        
        # Sort and select top pre_nms
        if pre_nms_topk < len(objectness):
            topk_idx = torch.topk(objectness, pre_nms_topk).indices
            proposal_boxes = proposal_boxes[topk_idx]
            objectness = objectness[topk_idx]
        
        # NMS
        keep_idx = batched_nms(proposal_boxes, objectness, torch.zeros(len(objectness), dtype=torch.long, device=objectness.device), nms_thresh)
        
        # Select top post_nms
        if post_nms_topk < len(keep_idx):
            keep_idx = keep_idx[:post_nms_topk]
        
        proposal_boxes = proposal_boxes[keep_idx]
        objectness = objectness[keep_idx]
        
        inst = Instances(img_size)
        inst.proposal_boxes = Boxes(proposal_boxes)
        inst.objectness_logits = objectness
        proposals.append(inst)
    
    return proposals
