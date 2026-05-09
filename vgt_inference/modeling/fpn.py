from __future__ import annotations

import torch.nn.functional as F
from torch import nn

from .backbone import Backbone, ShapeSpec


class LastLevelMaxPool(nn.Module):
    in_feature = "p5"
    num_levels = 1

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class FPN(Backbone):
    def __init__(
        self,
        bottom_up: Backbone,
        in_features: list[str],
        out_channels: int,
        norm: str = "",
        top_block: nn.Module | None = None,
        fuse_type: str = "sum",
    ) -> None:
        super().__init__()
        if fuse_type not in {"sum", "avg"}:
            raise ValueError(f"Unsupported FPN fuse_type: {fuse_type}")
        self.bottom_up = bottom_up
        self.in_features = in_features
        self.top_block = top_block
        self._fuse_type = fuse_type
        self._out_features = [f"p{int(bottom_up.output_shape()[name].stride).bit_length() - 1}" for name in in_features]
        if top_block is not None:
            last_stage = int(self._out_features[-1][1:])
            self._out_features.extend([f"p{stage}" for stage in range(last_stage + 1, last_stage + 1 + top_block.num_levels)])

        lateral_convs = []
        output_convs = []
        for in_feature in reversed(in_features):
            in_channels = bottom_up.output_shape()[in_feature].channels
            lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            output_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.output_convs = nn.ModuleList(output_convs)

        self._out_feature_strides = {
            out_feature: bottom_up.output_shape()[in_feature].stride
            for out_feature, in_feature in zip(self._out_features[: len(in_features)], in_features)
        }
        self._out_feature_channels = {out_feature: out_channels for out_feature in self._out_features}
        if top_block is not None:
            stride = self._out_feature_strides[self._out_features[len(in_features) - 1]]
            for out_feature in self._out_features[len(in_features) :]:
                stride *= 2
                self._out_feature_strides[out_feature] = stride

        # detectron2 compatibility: size_divisibility = stride of coarsest in_feature level (p5)
        self._size_divisibility = self._out_feature_strides[self._out_features[len(in_features) - 1]]

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        results.append(self.output_convs[0](prev_features))

        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            if idx == 0:
                continue
            features = bottom_up_features[self.in_features[-idx - 1]]
            # Interpolate to exact spatial size of lateral features to handle odd dimensions
            _, _, h, w = features.shape
            top_down_features = F.interpolate(prev_features, size=(h, w), mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        return {feature: result for feature, result in zip(self._out_features, results)}

    def output_shape(self) -> dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
            for name in self._out_features
        }
