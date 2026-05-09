from __future__ import annotations

import torch
from torch import nn

from vgt_inference.core.structures import Boxes, Instances


class GeneralizedRCNN(nn.Module):
    def __init__(
        self,
        *,
        backbone,
        proposal_generator,
        roi_heads,
        pixel_mean,
        pixel_std,
        input_format="BGR",
        vis_period=0,
    ):
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), persistent=False)
        self.input_format = input_format
        self.vis_period = vis_period

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return x.to(self.device)

    def preprocess_image(self, batched_inputs):
        from .imagelist import ImageList

        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            results.append({"instances": r})
        return results


def detector_postprocess(results: Instances, output_height: int, output_width: int):
    scale_x = output_width / results.image_size[1]
    scale_y = output_height / results.image_size[0]
    results = results.to(torch.device("cpu"))

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None

    if output_boxes is not None:
        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

    return results
