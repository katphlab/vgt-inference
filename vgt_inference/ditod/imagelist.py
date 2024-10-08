from typing import Any

import torch
from detectron2.layers.wrappers import move_device_like, shapes_to_tensor
from torch import Tensor, device
from torch.nn import functional as F


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    """

    def __init__(
        self,
        tensor: Tensor,
        padding_maks: Tensor,
        image_sizes: list[tuple[int, int]] | list[Tensor],
    ) -> None:
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.image_sizes = image_sizes
        self.padding_mask = padding_maks

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.padding_mask, self.image_sizes)

    @property
    def device(self) -> device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
        tensors: list[Tensor],
        size_divisibility: int = 0,
        pad_value: float = 0.0,
        padding_constraints: dict[str, int] | None = None,
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad.
            padding_constraints (optional[Dict]): If given, it would follow the format as
                {"size_divisibility": int, "square_size": int}, where `size_divisibility` will
                overwrite the above one if presented and `square_size` indicates the
                square padding size if `square_size` > 0.
        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [shapes_to_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if padding_constraints is not None:
            square_size = padding_constraints.get("square_size", 0)
            if square_size > 0:
                # pad to square.
                max_size[0] = max_size[1] = square_size
            if "size_divisibility" in padding_constraints:
                size_divisibility = padding_constraints["size_divisibility"]
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)).div(
                stride, rounding_mode="floor"
            ) * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: list[int] = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [
                0,
                max_size[-1] - image_size[1],
                0,
                max_size[-2] - image_size[0],
            ]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(
                0
            )
            attention_masks = create_attention_mask(tensors, max_size)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            device = (
                None
                if torch.jit.is_scripting()
                else ("cpu" if torch.jit.is_tracing() else None)
            )
            batched_imgs = tensors[0].new_full(batch_shape, pad_value, device=device)
            batched_imgs = move_device_like(batched_imgs, tensors[0])
            for i, img in enumerate(tensors):
                # Use `batched_imgs` directly instead of `img, pad_img = zip(tensors, batched_imgs)`
                # Tracing mode cannot capture `copy_()` of temporary locals
                batched_imgs[i, ..., : img.shape[-2], : img.shape[-1]].copy_(img)
            batched_imgs = batched_imgs.contiguous()
            attention_masks = create_attention_mask(tensors, max_size)
        return ImageList(batched_imgs, attention_masks, image_sizes)


def create_attention_mask(
    tensors: list[Tensor], max_size: list[int], patch_size: int = 16
) -> Tensor:
    attention_masks = []
    for img in tensors:
        orig_size = img.shape[-2:]  # (H, W)
        mask = torch.ones(orig_size, dtype=torch.float32, device=img.device)
        pad_size = (0, max_size[-1] - orig_size[1], 0, max_size[-2] - orig_size[0])
        padded_mask = F.pad(mask, pad_size, value=0)
        pooled_mask = F.avg_pool2d(
            padded_mask.unsqueeze(0).unsqueeze(0), kernel_size=patch_size
        ).squeeze()
        pooled_mask = (pooled_mask > 0).float()
        attention_mask_1d = pooled_mask.view(-1)
        attention_mask_1d = torch.cat(
            [torch.tensor([1.0], device=img.device), attention_mask_1d]
        )
        large_neg_val = -1e6
        attention_mask_1d = (1 - attention_mask_1d) * large_neg_val
        attention_masks.append(attention_mask_1d)
    attention_mask = torch.stack(attention_masks, dim=0)
    return attention_mask
