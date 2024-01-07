from detectron2.config import get_cfg

from .ditod import add_vit_config
from .ditod.tokenization_bros import BrosTokenizer
from .predictor import DefaultPredictor


class VGTPredictor:
    def __init__(
        self, config_path: str, model_weights_path: str, device: str, hf_token: str
    ) -> None:
        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file(config_path)

        cfg.MODEL.DEVICE = device

        self.cfg = cfg
        self.hf_token = hf_token
        self.model_weights_path = model_weights_path

        self.tokenizer = None
        self.predictor = None

    def load(self) -> None:
        self.tokenizer = BrosTokenizer.from_pretrained(
            "thewalnutaisg/bros-base-uncased", token=self.hf_token
        )
        self.predictor = DefaultPredictor(self.cfg, self.model_weights_path)
