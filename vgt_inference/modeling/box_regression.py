from __future__ import annotations

import math

import torch


class Box2BoxTransform:
    def __init__(self, weights: tuple[float, float, float, float]) -> None:
        self.weights = weights

    def apply_deltas(self, deltas, boxes):
        """
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4) or (N, M, 4).
            boxes (Tensor): boxes to transform, of shape (N, 4) or (N, M, 4).
        """
        boxes = boxes.to(deltas.dtype)
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh

        dw = torch.clamp(dw, max=math.log(1000 / 16))
        dh = torch.clamp(dh, max=math.log(1000 / 16))

        pred_ctr_x = dx * widths[..., None] + ctr_x[..., None]
        pred_ctr_y = dy * heights[..., None] + ctr_y[..., None]
        pred_w = torch.exp(dw) * widths[..., None]
        pred_h = torch.exp(dh) * heights[..., None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[..., 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[..., 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[..., 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[..., 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes

    def get_deltas(self, src_boxes, target_boxes):
        src_widths = src_boxes[:, 2] - src_boxes[:, 0]
        src_heights = src_boxes[:, 3] - src_boxes[:, 1]
        src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

        target_widths = target_boxes[:, 2] - target_boxes[:, 0]
        target_heights = target_boxes[:, 3] - target_boxes[:, 1]
        target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=1)
        return deltas
