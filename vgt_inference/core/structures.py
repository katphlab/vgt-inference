from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
import torch


class BoxMode(IntEnum):
    XYXY_ABS = 0
    XYWH_ABS = 1

    @staticmethod
    def convert(box: np.ndarray | torch.Tensor | list, from_mode: "BoxMode", to_mode: "BoxMode"):
        if from_mode == to_mode:
            return box.copy() if isinstance(box, np.ndarray) else box.clone() if torch.is_tensor(box) else list(box)
        if from_mode != BoxMode.XYWH_ABS or to_mode != BoxMode.XYXY_ABS:
            raise NotImplementedError(f"Unsupported BoxMode conversion: {from_mode} -> {to_mode}")

        converted = box.copy() if isinstance(box, np.ndarray) else box.clone() if torch.is_tensor(box) else np.asarray(box).copy()
        converted[..., 2] = converted[..., 0] + converted[..., 2]
        converted[..., 3] = converted[..., 1] + converted[..., 3]
        return converted


class Boxes:
    def __init__(self, tensor: torch.Tensor | np.ndarray | list) -> None:
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, 4))
        if tensor.dim() != 2 or tensor.size(-1) != 4:
            raise ValueError("Boxes tensor must have shape (N, 4)")
        self.tensor = tensor

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, item) -> "Boxes":
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, 4))
        return Boxes(self.tensor[item])

    def clone(self) -> "Boxes":
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device | str) -> "Boxes":
        return Boxes(self.tensor.to(device))

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0] *= scale_x
        self.tensor[:, 2] *= scale_x
        self.tensor[:, 1] *= scale_y
        self.tensor[:, 3] *= scale_y

    def clip(self, box_size: tuple[int, int]) -> None:
        height, width = box_size
        self.tensor[:, 0].clamp_(min=0, max=width)
        self.tensor[:, 2].clamp_(min=0, max=width)
        self.tensor[:, 1].clamp_(min=0, max=height)
        self.tensor[:, 3].clamp_(min=0, max=height)

    def area(self) -> torch.Tensor:
        return (self.tensor[:, 2] - self.tensor[:, 0]) * (self.tensor[:, 3] - self.tensor[:, 1])

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        widths = self.tensor[:, 2] - self.tensor[:, 0]
        heights = self.tensor[:, 3] - self.tensor[:, 1]
        return (widths > threshold) & (heights > threshold)

    @staticmethod
    def cat(boxes_list: list["Boxes"]) -> "Boxes":
        if not boxes_list:
            return Boxes(torch.empty(0, 4))
        return Boxes(torch.cat([boxes.tensor for boxes in boxes_list], dim=0))


class Instances:
    def __init__(self, image_size: tuple[int, int], **kwargs: Any) -> None:
        object.__setattr__(self, "_image_size", tuple(image_size))
        object.__setattr__(self, "_fields", {})
        for key, value in kwargs.items():
            self.set(key, value)

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def __len__(self) -> int:
        if not self._fields:
            raise NotImplementedError("Empty Instances does not support len()")
        first_value = next(iter(self._fields.values()))
        return len(first_value)

    def __getattr__(self, name: str) -> Any:
        # Use object.__getattribute__ to avoid recursion for _fields
        try:
            _fields = object.__getattribute__(self, "_fields")
        except AttributeError:
            raise AttributeError(name)
        if name in _fields:
            return _fields[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self.set(name, value)

    def __getitem__(self, item) -> "Instances":
        if isinstance(item, int):
            item = slice(item, item + 1)
        instances = Instances(self.image_size)
        for key, value in self._fields.items():
            instances.set(key, value[item])
        return instances

    def set(self, name: str, value: Any) -> None:
        if self._fields and len(value) != len(self):
            raise ValueError(f"Field '{name}' has length {len(value)} but Instances has length {len(self)}")
        self._fields[name] = value

    def get(self, name: str) -> Any:
        return self._fields[name]

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]

    def get_fields(self) -> dict[str, Any]:
        return self._fields

    def to(self, device: torch.device | str) -> "Instances":
        instances = Instances(self.image_size)
        for key, value in self._fields.items():
            instances.set(key, value.to(device) if hasattr(value, "to") else value)
        return instances

    @staticmethod
    def cat(instances_list: list["Instances"]) -> "Instances":
        if not instances_list:
            raise ValueError("Instances.cat requires a non-empty list")
        image_size = instances_list[0].image_size
        if any(instances.image_size != image_size for instances in instances_list):
            raise ValueError("All Instances must have the same image_size")

        output = Instances(image_size)
        for key in instances_list[0]._fields:
            values = [instances._fields[key] for instances in instances_list]
            first = values[0]
            if hasattr(type(first), "cat"):
                output.set(key, type(first).cat(values))
            elif torch.is_tensor(first):
                output.set(key, torch.cat(values, dim=0))
            else:
                merged = []
                for value in values:
                    merged.extend(value)
                output.set(key, merged)
        return output
