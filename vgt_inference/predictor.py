import torch
from numpy import ndarray

from vgt_inference.core.config import CfgNode
from vgt_inference.core.structures import BoxMode
from vgt_inference.core.transforms import ResizeShortestEdge, apply_transform_gens, transform_instance_annotations
from vgt_inference.modeling.build import build_model
from vgt_inference.modeling.checkpoint import Checkpointer


class DefaultPredictor:
    def __init__(self, cfg: CfgNode, model_weights_path: str = None) -> None:
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = Checkpointer(self.model)
        checkpointer.load(model_weights_path)

        self.aug = ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(
        self, image_list: list[ndarray], grid_data_list: list[dict[str, list]]
    ) -> list:
        with torch.no_grad():
            dataset_list = []
            for original_image, grid_data in zip(image_list, grid_data_list):
                height, width = original_image.shape[:2]
                image, transforms = apply_transform_gens([self.aug], original_image)

                image_shape = image.shape[:2]
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input_ids = grid_data["input_ids"]
                bbox_subword_list = grid_data["bbox_subword_list"]

                bbox = []
                for bbox_per_subword in bbox_subword_list:
                    text_word = {}
                    text_word["bbox"] = bbox_per_subword
                    text_word["bbox_mode"] = BoxMode.XYXY_ABS
                    transform_instance_annotations(
                        text_word, transforms, image_shape
                    )
                    bbox.append(text_word["bbox"])

                dataset_dict = {}
                dataset_dict["input_ids"] = input_ids
                dataset_dict["bbox"] = bbox
                dataset_dict["image"] = image
                dataset_dict["height"] = height
                dataset_dict["width"] = width
                dataset_list.append(dataset_dict)

            predictions = self.model(dataset_list)
            return predictions
