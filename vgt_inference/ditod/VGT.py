# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

from vgt_inference.core.structures import Instances
from vgt_inference.modeling.meta_arch import GeneralizedRCNN

from .imagelist import ImageList
from .Wordnn_embedding import WordnnEmbedding

__all__ = ["VGT"]


def torch_memory(device, tag="") -> None:
    # Checks and prints GPU memory
    print(tag, f"{torch.cuda.memory_allocated(device)/1024/1024:.2f} MB USED")
    print(tag, f"{torch.cuda.memory_reserved(device)/1024/1024:.2f} MB RESERVED")
    print(tag, f"{torch.cuda.max_memory_allocated(device)/1024/1024:.2f} MB USED MAX")
    print(
        tag, f"{torch.cuda.max_memory_reserved(device)/1024/1024:.2f} MB RESERVED MAX"
    )
    print("")


class VGT(GeneralizedRCNN):
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
        vocab_size: int = 30552,
        hidden_size: int = 768,
        embedding_dim: int = 64,
        bros_embedding_path: str = "",
        use_pretrain_weight: bool = True,
        use_UNK_text: bool = False,
    ) -> None:
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.Wordgrid_embedding = WordnnEmbedding(
            vocab_size,
            hidden_size,
            embedding_dim,
            bros_embedding_path,
            use_pretrain_weight,
            use_UNK_text,
        )

    def forward(self, batched_inputs: list[dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        chargrid = self.Wordgrid_embedding(images.tensor, batched_inputs)
        features = self.backbone(
            images.tensor, chargrid, attention_mask=images.padding_mask
        )

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses

    def inference(
        self,
        batched_inputs: list[dict[str, torch.Tensor]],
        detected_instances: list[Instances] | None = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)

        chargrid = self.Wordgrid_embedding(images.tensor, batched_inputs)
        features = self.backbone(
            images.tensor, chargrid, attention_mask=images.padding_mask
        )

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results

    def preprocess_image(
        self, batched_inputs: list[dict[str, torch.Tensor]]
    ) -> ImageList:
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
        )
        return images
