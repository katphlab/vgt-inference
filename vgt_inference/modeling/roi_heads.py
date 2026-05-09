from __future__ import annotations

import math

import torch
from torch import nn
import torchvision

from vgt_inference.core.structures import Boxes, Instances
from vgt_inference.modeling.box_regression import Box2BoxTransform
from vgt_inference.modeling.nms import batched_nms


class FastRCNNOutputLayers(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        cls_agnostic_bbox_reg: bool = False,
        box_dim: int = 4,
        test_score_thresh: float = 0.05,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas


class FastRCNNConvFCHead(nn.Module):
    def __init__(self, input_shape, fc_dim=1024, num_fc=2):
        super().__init__()
        in_channels = input_shape.channels
        self.flatten = nn.Flatten()
        self.fcs = nn.ModuleList()
        in_dim = in_channels * input_shape.width * input_shape.height
        for k in range(num_fc):
            fc = nn.Linear(in_dim, fc_dim)
            self.fcs.append(fc)
            in_dim = fc_dim
        
        for layer in self.fcs:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)
        
        self._output_size = fc_dim

    @property
    def output_shape(self):
        return type("Shape", (), {"channels": self._output_size})()

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.fcs:
            x = torch.relu(layer(x))
        return x


class CascadeROIHeads(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        batch_size_per_image: int,
        positive_fraction: float,
        proposal_append_gt: bool,
        in_features: list[str],
        pooler_resolution: int = 7,
        pooler_scales=None,
        pooler_sampling_ratio: int = 0,
        pooler_type: str = "ROIAlignV2",
        box_head=None,
        box_predictor=None,
        mask_head=None,
        train_on_pred_boxes: bool = False,
        cascade_bbox_reg_weights=[(10.0, 10.0, 5.0, 5.0), (20.0, 20.0, 10.0, 10.0), (30.0, 30.0, 15.0, 15.0)],
        cascade_ious=[0.5, 0.6, 0.7],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.proposal_append_gt = proposal_append_gt
        self.in_features = in_features
        self.train_on_pred_boxes = train_on_pred_boxes
        self.cascade_bbox_reg_weights = cascade_bbox_reg_weights
        self.cascade_ious = cascade_ious

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales if pooler_scales is not None else [0.25, 0.125, 0.0625, 0.03125],
            sampling_ratio=pooler_sampling_ratio,
            pooler_type=pooler_type,
        )
        
        # Create cascade stages
        import copy
        num_stages = len(cascade_bbox_reg_weights)
        self.box_head = nn.ModuleList()
        self.box_predictor = nn.ModuleList()
        for k in range(num_stages):
            self.box_head.append(copy.deepcopy(box_head))
            self.box_predictor.append(copy.deepcopy(box_predictor))

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            return self._forward_train(features, proposals, targets)
        else:
            return self._forward_inference(features, proposals)

    def _forward_train(self, features, proposals, targets):
        # Simplified training - not used for inference migration
        pred_instances = proposals
        return pred_instances, {}

    def _forward_inference(self, features, proposals):
        """Inference following detectron2's Cascade R-CNN implementation."""
        features = [features[f] for f in self.in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        
        for k in range(len(self.cascade_bbox_reg_weights)):
            if k > 0:
                # Create new proposals from refined boxes
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
            
            # Run this stage
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self._predict_boxes(predictions, proposals, k)
            head_outputs.append((self.box_predictor[k], predictions, proposals))
        
        # Collect per-stage probabilities
        scores_per_stage = [
            self._predict_probs(predictor, predictions, proposals)
            for predictor, predictions, proposals in head_outputs
        ]
        
        # Average scores across stages (detectron2 behavior)
        averaged_scores = [
            sum(list(scores_per_image)) * (1.0 / len(self.cascade_bbox_reg_weights))
            for scores_per_image in zip(*scores_per_stage)
        ]
        
        # Use boxes from the last head
        predictor, predictions, proposals = head_outputs[-1]
        boxes = self._predict_boxes(predictions, proposals, len(self.cascade_bbox_reg_weights) - 1)
        
        # Run fast_rcnn_inference
        results = []
        for boxes_per_image, scores_per_image, image_size in zip(boxes, averaged_scores, image_sizes):
            result, _ = fast_rcnn_inference_single_image(
                boxes_per_image,
                scores_per_image,
                image_size,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            results.append(result)
        
        return results, {}
    
    def _run_stage(self, features, proposals, stage):
        """Run a single cascade stage.
        
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads in same order as scales.
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](box_features)
    
    def _predict_boxes(self, predictions, proposals, stage):
        """Predict refined boxes from deltas."""
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        
        box2box_transform = Box2BoxTransform(weights=self.cascade_bbox_reg_weights[stage])
        pred_boxes = box2box_transform.apply_deltas(proposal_deltas, proposal_boxes)
        
        return pred_boxes.split(num_prop_per_image)
    
    def _predict_probs(self, predictor, predictions, proposals):
        """Predict class probabilities."""
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = torch.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
    
    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """Create new proposals from refined boxes for next cascade stage."""
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

    def forward_with_given_boxes(self, features, detected_instances):
        # For keypoint/mask prediction on given boxes
        return detected_instances


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape,
    score_thresh,
    nms_thresh,
    topk_per_image,
):
    """
    Single-image inference matching detectron2's fast_rcnn_inference_single_image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    
    scores = scores[:, :-1]  # Remove background class
    num_bbox_reg_classes = boxes.shape[1] // 4
    
    # Clip boxes to image
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)
    
    # Filter by score threshold
    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero(as_tuple=False)
    
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    
    # Apply NMS per class
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def assign_boxes_to_levels(box_lists, min_level, max_level, canonical_box_size, canonical_level):
    """
    Match detectron2's assign_boxes_to_levels:
    k = k0 + log2(sqrt(area) / canonical_box_size)
    then floor(), then clamp.
    """
    from vgt_inference.core.structures import Boxes
    box_sizes = torch.sqrt(torch.cat([b.area() for b in box_lists]))
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + 1e-8)
    )
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def convert_boxes_to_pooler_format(box_lists):
    """
    Match detectron2's convert_boxes_to_pooler_format:
    (batch_idx, x0, y0, x1, y1)
    """
    boxes = torch.cat([b.tensor for b in box_lists], dim=0)
    sizes = torch.tensor([len(b) for b in box_lists], dtype=torch.long, device=boxes.device)
    indices = torch.repeat_interleave(
        torch.arange(len(sizes), dtype=torch.long, device=boxes.device), sizes
    )
    return torch.cat([indices[:, None].to(boxes.dtype), boxes], dim=1)


class ROIPooler(nn.Module):
    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        self.output_size = output_size

        if pooler_type == "ROIAlign":
            from torchvision.ops import RoIAlign as _ROIAlign
            self.level_poolers = nn.ModuleList(
                _ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            from torchvision.ops import RoIAlign as _ROIAlign
            self.level_poolers = nn.ModuleList(
                _ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            from torchvision.ops import RoIPool as _RoIPool
            self.level_poolers = nn.ModuleList(
                _RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert len(scales) == self.max_level - self.min_level + 1, \
            "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x, box_lists):
        """
        Args:
            x: list[Tensor] of feature maps (same order as scales used in __init__)
            box_lists: list[Boxes] per image
        """
        num_level_assignments = len(self.level_poolers)
        assert len(x) == num_level_assignments
        assert len(box_lists) == x[0].size(0)
        if len(box_lists) == 0:
            return torch.empty(
                0, x[0].shape[1], self.output_size[0], self.output_size[1],
                device=x[0].device, dtype=x[0].dtype,
            )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists,
            self.min_level,
            self.max_level,
            self.canonical_box_size,
            self.canonical_level,
        )

        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        output = torch.zeros(
            pooler_fmt_boxes.shape[0],
            num_channels,
            output_size,
            output_size,
            device=x[0].device,
            dtype=x[0].dtype,
        )

        for level, pooler in enumerate(self.level_poolers):
            inds = (level_assignments == level).nonzero(as_tuple=False).squeeze(1)
            if inds.numel() == 0:
                continue
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x[level], pooler_fmt_boxes_level)

        return output
