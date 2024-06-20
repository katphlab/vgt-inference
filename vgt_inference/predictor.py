import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
from detectron2.structures import BoxMode
from detectron2.structures.instances import Instances
from numpy import ndarray

from .ditod import aug


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg: CfgNode, model_weights_path: str = None) -> None:
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(model_weights_path)

        self.aug = aug.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(
        self, image_list: list[ndarray], grid_data_list: list[dict[str, list]]
    ) -> list:
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            dataset_list = []
            for original_image, grid_data in zip(image_list, grid_data_list):
                height, width = original_image.shape[:2]
                image, transforms = T.apply_transform_gens([self.aug], original_image)

                # add grid
                image_shape = image.shape[:2]  # h, w
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input_ids = grid_data["input_ids"]
                bbox_subword_list = grid_data["bbox_subword_list"]

                # word bbox
                bbox = []
                for bbox_per_subword in bbox_subword_list:
                    text_word = {}
                    text_word["bbox"] = bbox_per_subword
                    text_word["bbox_mode"] = BoxMode.XYWH_ABS
                    utils.transform_instance_annotations(
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
