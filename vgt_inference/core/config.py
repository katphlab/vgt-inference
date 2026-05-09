from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import yaml


class CfgNode(dict):
    def __init__(self, init_dict: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_frozen", False)
        super().__init__()
        for key, value in (init_dict or {}).items():
            self[key] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self[name] = value

    def __setitem__(self, key: str, value: Any) -> None:
        if self._frozen:
            raise AttributeError("Attempted to modify a frozen CfgNode")
        super().__setitem__(key, self._wrap(value))

    def clone(self) -> "CfgNode":
        return copy.deepcopy(self)

    def freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)
        for value in self.values():
            if isinstance(value, CfgNode):
                value.freeze()

    def defrost(self) -> None:
        object.__setattr__(self, "_frozen", False)
        for value in self.values():
            if isinstance(value, CfgNode):
                value.defrost()

    def merge_from_file(self, cfg_filename: str) -> None:
        config_path = Path(cfg_filename)
        data = _load_yaml_with_base(config_path)
        self.merge_from_other_cfg(CfgNode(data))

    def merge_from_other_cfg(self, other_cfg: "CfgNode") -> None:
        _merge_a_into_b(other_cfg, self)

    def merge_from_list(self, cfg_list: list[str]) -> None:
        if len(cfg_list) % 2 != 0:
            raise ValueError("Config override list must contain KEY VALUE pairs")
        for dotted_key, raw_value in zip(cfg_list[0::2], cfg_list[1::2]):
            node = self
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                if part not in node:
                    node[part] = CfgNode()
                node = node[part]
                if not isinstance(node, CfgNode):
                    raise KeyError(f"Cannot merge into non-config key: {dotted_key}")
            node[parts[-1]] = _decode_value(raw_value)

    def to_dict(self) -> dict[str, Any]:
        return {key: _unwrap(value) for key, value in self.items()}

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, CfgNode):
            return value
        if isinstance(value, dict):
            return CfgNode(value)
        if isinstance(value, list):
            return [CfgNode._wrap(item) for item in value]
        return value


def get_cfg() -> CfgNode:
    return CfgNode(
        {
            "VERSION": 2,
            "SEED": -1,
            "INPUT": {
                "FORMAT": "BGR",
                "MIN_SIZE_TEST": 800,
                "MAX_SIZE_TEST": 1333,
                "MIN_SIZE_TRAIN": (),
                "MAX_SIZE_TRAIN": 1333,
                "CROP": {"ENABLED": False, "TYPE": "absolute_range", "SIZE": ()},
            },
            "MODEL": {
                "DEVICE": "cpu",
                "WEIGHTS": "",
                "MASK_ON": False,
                "META_ARCHITECTURE": "GeneralizedRCNN",
                "PIXEL_MEAN": [103.53, 116.28, 123.675],
                "PIXEL_STD": [1.0, 1.0, 1.0],
                "BACKBONE": {"NAME": "", "FREEZE_AT": 0},
                "FPN": {
                    "IN_FEATURES": [],
                    "OUT_CHANNELS": 256,
                    "NORM": "",
                    "FUSE_TYPE": "sum",
                },
                "VIT": {
                    "NAME": "",
                    "OUT_FEATURES": ["layer3", "layer5", "layer7", "layer11"],
                    "IMG_SIZE": [224, 224],
                    "POS_TYPE": "shared_rel",
                    "MERGE_TYPE": "Sum",
                    "DROP_PATH": 0.0,
                    "MODEL_KWARGS": "{}",
                },
                "WORDGRID": {
                    "VOCAB_SIZE": 30552,
                    "EMBEDDING_DIM": 64,
                    "MODEL_PATH": "",
                    "HIDDEN_SIZE": 768,
                    "USE_PRETRAIN_WEIGHT": True,
                    "USE_UNK_TEXT": False,
                },
                "RPN": {
                    "IN_FEATURES": [],
                    "PRE_NMS_TOPK_TRAIN": 12000,
                    "PRE_NMS_TOPK_TEST": 6000,
                    "POST_NMS_TOPK_TRAIN": 2000,
                    "POST_NMS_TOPK_TEST": 1000,
                    "NMS_THRESH": 0.7,
                    "BBOX_REG_WEIGHTS": (1.0, 1.0, 1.0, 1.0),
                    "MIN_SIZE": 0.0,
                    "CONV_DIMS": [-1],
                },
                "ANCHOR_GENERATOR": {
                    "NAME": "DefaultAnchorGenerator",
                    "SIZES": [[32], [64], [128], [256], [512]],
                    "ASPECT_RATIOS": [[0.5, 1.0, 2.0]],
                    "ANGLES": [[-90, 0, 90]],
                    "OFFSET": 0.0,
                },
                "PROPOSAL_GENERATOR": {"NAME": "RPN", "MIN_SIZE": 0},
                "ROI_HEADS": {
                    "NAME": "StandardROIHeads",
                    "IN_FEATURES": [],
                    "NUM_CLASSES": 80,
                    "SCORE_THRESH_TEST": 0.05,
                    "NMS_THRESH_TEST": 0.5,
                    "BATCH_SIZE_PER_IMAGE": 512,
                    "POSITIVE_FRACTION": 0.25,
                    "PROPOSAL_APPEND_GT": True,
                },
                "ROI_BOX_HEAD": {
                    "NAME": "FastRCNNConvFCHead",
                    "POOLER_RESOLUTION": 7,
                    "POOLER_SAMPLING_RATIO": 0,
                    "POOLER_TYPE": "ROIAlignV2",
                    "NUM_FC": 2,
                    "FC_DIM": 1024,
                    "NUM_CONV": 0,
                    "CONV_DIM": 256,
                    "CLS_AGNOSTIC_BBOX_REG": False,
                    "BBOX_REG_WEIGHTS": (10.0, 10.0, 5.0, 5.0),
                    "BBOX_REG_LOSS_TYPE": "smooth_l1",
                    "SMOOTH_L1_BETA": 0.0,
                },
                "ROI_MASK_HEAD": {"NAME": "MaskRCNNConvUpsampleHead", "NUM_CONV": 4, "POOLER_RESOLUTION": 14},
            },
            "TEST": {"EVAL_PERIOD": 0, "DETECTIONS_PER_IMAGE": 100},
            "DATASETS": {"TRAIN": (), "TEST": ()},
            "DATALOADER": {"NUM_WORKERS": 0, "FILTER_EMPTY_ANNOTATIONS": True},
            "SOLVER": {
                "OPTIMIZER": "ADAMW",
                "BACKBONE_MULTIPLIER": 1.0,
                "BASE_LR": 0.001,
                "WARMUP_ITERS": 0,
                "IMS_PER_BATCH": 1,
                "MAX_ITER": 0,
                "STEPS": (),
                "AMP": {"ENABLED": False},
                "CLIP_GRADIENTS": {"ENABLED": False, "CLIP_TYPE": "full_model", "CLIP_VALUE": 1.0, "NORM_TYPE": 2.0},
            },
            "AUG": {"DETR": False},
        }
    )


def _merge_a_into_b(source: CfgNode, dest: CfgNode) -> None:
    for key, value in source.items():
        if isinstance(value, CfgNode) and isinstance(dest.get(key), CfgNode):
            _merge_a_into_b(value, dest[key])
        else:
            dest[key] = value


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data = _decode_container(data)
    base_value = data.pop("_BASE_", None)
    if not base_value:
        return data

    base_paths = base_value if isinstance(base_value, list) else [base_value]
    merged: dict[str, Any] = {}
    for base_path in base_paths:
        base_data = _load_yaml_with_base((path.parent / base_path).resolve())
        _merge_plain_dict(base_data, merged)
    _merge_plain_dict(data, merged)
    return merged


def _merge_plain_dict(source: dict[str, Any], dest: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            _merge_plain_dict(value, dest[key])
        else:
            dest[key] = copy.deepcopy(value)


def _decode_container(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _decode_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_container(item) for item in value]
    return _decode_value(value)


def _decode_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        return ast.literal_eval(stripped)
    try:
        loaded = yaml.safe_load(stripped)
    except yaml.YAMLError:
        return value
    return value if loaded is None else loaded


def _unwrap(value: Any) -> Any:
    if isinstance(value, CfgNode):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value
