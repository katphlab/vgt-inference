from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


class DefaultAnchorGenerator(nn.Module):
    def __init__(
        self,
        *,
        sizes: Sequence[Sequence[int]] = ((128, 256, 512),),
        aspect_ratios: Sequence[Sequence[float]] = ((0.5, 1.0, 2.0),),
        strides: Sequence[int] | None = None,
        offset: float = 0.0,
    ):
        super().__init__()
        self.sizes = [list(s) for s in sizes]
        self.aspect_ratios = [list(a) for a in aspect_ratios]
        # Broadcast aspect_ratios to match the number of sizes (detectron2 behavior)
        if len(self.aspect_ratios) == 1 and len(self.sizes) > 1:
            self.aspect_ratios = [self.aspect_ratios[0]] * len(self.sizes)
        self.offset = offset
        self.strides = strides if strides is not None else [2 ** i for i in range(len(self.sizes))]
        self.num_features = len(self.strides)
        self.cell_anchors = self._calculate_anchors()

    def _calculate_anchors(self):
        cell_anchors = []
        for s, a in zip(self.sizes, self.aspect_ratios):
            anchors = []
            for size in s:
                area = size ** 2
                for aspect in a:
                    w = math.sqrt(area / aspect)
                    h = aspect * w
                    anchors.append([-w / 2, -h / 2, w / 2, h / 2])
            cell_anchors.append(torch.tensor(anchors, dtype=torch.float32))
        return cell_anchors

    def forward(self, features):
        """
        Args:
            features: list[Tensor] of feature maps for each level, shape (N, C, H, W)
        
        Returns:
            list[list[Tensor]]: For each level, a list of anchor tensors per image.
        """
        anchors = []
        for i, (feature, stride, cell_anchor) in enumerate(
            zip(features, self.strides, self.cell_anchors)
        ):
            N, _, h, w = feature.shape
            device = feature.device
            cell_anchor = cell_anchor.to(device)
            shift_x = torch.arange(w, device=device) * stride + self.offset
            shift_y = torch.arange(h, device=device) * stride + self.offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shifts = torch.stack([
                shift_x.reshape(-1), shift_y.reshape(-1),
                shift_x.reshape(-1), shift_y.reshape(-1)
            ], dim=1)
            # Generate anchors for one image
            level_anchors_one = (shifts.view(-1, 1, 4) + cell_anchor.view(1, -1, 4)).reshape(-1, 4)
            # Repeat for all images in the batch
            level_anchors = level_anchors_one.repeat(N, 1)
            num_anchors_per_image = level_anchors_one.shape[0]
            # Split per image
            anchors_per_image = level_anchors.split(num_anchors_per_image)
            anchors.append(list(anchors_per_image))
        return anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]
