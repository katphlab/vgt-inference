from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from vgt_inference.core.registry import Registry


@dataclass
class ShapeSpec:
    channels: int | None = None
    height: int | None = None
    width: int | None = None
    stride: int | None = None


class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._size_divisibility = 0

    @property
    def size_divisibility(self) -> int:
        return self._size_divisibility

    def output_shape(self) -> dict[str, ShapeSpec]:
        raise NotImplementedError


BACKBONE_REGISTRY = Registry("BACKBONE")
