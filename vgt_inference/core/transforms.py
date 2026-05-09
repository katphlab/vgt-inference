from __future__ import annotations

import sys
from typing import Sequence

import numpy as np
from PIL import Image

from .structures import BoxMode


class Transform:
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return image

    def apply_box(self, boxes: np.ndarray) -> np.ndarray:
        return boxes


class NoOpTransform(Transform):
    pass


class ResizeTransform(Transform):
    def __init__(self, h: int, w: int, new_h: int, new_w: int, interp=Image.BILINEAR) -> None:
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == (self.new_h, self.new_w):
            return image
        pil_image = Image.fromarray(image)
        return np.asarray(pil_image.resize((self.new_w, self.new_h), self.interp))

    def apply_box(self, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        scale_x = self.new_w / self.w
        scale_y = self.new_h / self.h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        return boxes


class TransformList(Transform):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = list(transforms)

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            image = transform.apply_image(image)
        return image

    def apply_box(self, boxes: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            boxes = transform.apply_box(boxes)
        return boxes


class ResizeShortestEdge:
    def __init__(self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR) -> None:
        if sample_style not in {"range", "choice"}:
            raise ValueError(f"Unsupported sample_style: {sample_style}")
        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range and len(short_edge_length) != 2:
            raise ValueError("short_edge_length must contain two values for range sampling")
        self.short_edge_length = tuple(short_edge_length)
        self.max_size = max_size
        self.sample_style = sample_style
        self.interp = interp

    def get_transform(self, image: np.ndarray) -> Transform:
        h, w = image.shape[:2]
        if self.is_range:
            min_size = self.short_edge_length[0]
            max_size = self.short_edge_length[1]
            shortest_edge = min(h, w)
            size = min(max_size, max(min_size, shortest_edge))
        else:
            size = int(np.random.choice(self.short_edge_length))
        if size == 0:
            return NoOpTransform()
        new_h, new_w = self.get_output_shape(h, w, size, self.max_size)
        return ResizeTransform(h, w, new_h, new_w, self.interp)

    @staticmethod
    def get_output_shape(oldh: int, oldw: int, short_edge_length: int, max_size: int) -> tuple[int, int]:
        size = short_edge_length * 1.0
        scale = size / min(oldh, oldw)
        if oldh < oldw:
            newh, neww = size, scale * oldw
        else:
            newh, neww = scale * oldh, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh *= scale
            neww *= scale
        return int(newh + 0.5), int(neww + 0.5)


def apply_transform_gens(transform_gens: Sequence[ResizeShortestEdge], image: np.ndarray) -> tuple[np.ndarray, TransformList]:
    transforms = []
    for transform_gen in transform_gens:
        transform = transform_gen.get_transform(image)
        image = transform.apply_image(image)
        transforms.append(transform)
    return image, TransformList(transforms)


def transform_instance_annotations(annotation: dict, transforms: TransformList, image_size: tuple[int, int]) -> dict:
    bbox = BoxMode.convert(np.asarray([annotation["bbox"]], dtype=np.float32), annotation["bbox_mode"], BoxMode.XYXY_ABS)
    bbox = transforms.apply_box(bbox)
    height, width = image_size
    bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, width)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, height)
    annotation["bbox"] = bbox[0].tolist()
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation
