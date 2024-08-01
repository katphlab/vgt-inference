from detectron2.config import get_cfg

from .ditod import add_vit_config
from .ditod.tokenization_bros import BrosTokenizer
from .predictor import DefaultPredictor


class VGTPredictor:
    def __init__(
        self,
        config_path: str,
        model_weights_path: str,
        tokenizer_path: str,
        device: str,
    ) -> None:
        cfg = get_cfg()
        # As detectron2 checks config from original config, we set the MIN_SIZE_TEST to tuple for custom aug
        cfg.INPUT.MIN_SIZE_TEST = ()
        add_vit_config(cfg)
        cfg.merge_from_file(config_path)

        cfg.MODEL.DEVICE = device

        self.tokenizer = BrosTokenizer.from_pretrained(tokenizer_path)
        self.predictor = DefaultPredictor(cfg, model_weights_path)
