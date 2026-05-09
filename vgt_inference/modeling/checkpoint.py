from __future__ import annotations

import logging
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, model, save_dir="", *, save_to_disk=None):
        self.model = model
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

    def load(self, path: str | Path, checkpointables=None):
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        
        self._load_model(checkpoint)
        return {}

    def _load_model(self, checkpoint):
        model_state = self.model.state_dict()
        
        # Convert detectron2 checkpoint keys to local model keys
        converted_checkpoint = self._convert_checkpoint_keys(checkpoint, model_state)
        
        # Try to load matching keys
        loaded = {}
        missing = []
        unexpected = []
        
        for key in model_state:
            if key in converted_checkpoint:
                if converted_checkpoint[key].shape == model_state[key].shape:
                    loaded[key] = converted_checkpoint[key]
                else:
                    logger.warning(f"Shape mismatch for {key}: {converted_checkpoint[key].shape} vs {model_state[key].shape}")
                    missing.append(key)
            else:
                missing.append(key)
        
        for key in converted_checkpoint:
            if key not in model_state:
                unexpected.append(key)
        
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        
        self.model.load_state_dict(loaded, strict=False)
    
    @staticmethod
    def _convert_checkpoint_keys(checkpoint, model_state):
        """Map detectron2 checkpoint keys to local model keys."""
        converted = {}
        for key, value in checkpoint.items():
            converted_key = key
            
            # FPN lateral convs: backbone.fpn_lateral{stage} -> backbone.lateral_convs.{idx}
            # detectron2 stages: fpn_lateral2 (lowest/res2), fpn_lateral3, fpn_lateral4, fpn_lateral5 (highest/res5)
            # local lateral_convs are built in REVERSED order:
            #   lateral_convs[0] = highest level (res5/layer11)
            #   lateral_convs[3] = lowest level (res2/layer3)
            # So the mapping is: idx = (max_stage - stage) where max_stage = 5
            if "backbone.fpn_lateral" in key:
                stage = int(key.split("fpn_lateral")[1].split(".")[0])
                idx = 5 - stage  # reverse: 5->0, 4->1, 3->2, 2->3
                converted_key = key.replace(f"backbone.fpn_lateral{stage}", f"backbone.lateral_convs.{idx}")
            
            # FPN output convs: backbone.fpn_output{stage} -> backbone.output_convs.{idx}
            # Same reversed mapping as lateral_convs
            elif "backbone.fpn_output" in key:
                stage = int(key.split("fpn_output")[1].split(".")[0])
                idx = 5 - stage  # reverse: 5->0, 4->1, 3->2, 2->3
                converted_key = key.replace(f"backbone.fpn_output{stage}", f"backbone.output_convs.{idx}")
            
            # ROI box head FC layers: roi_heads.box_head.{stage}.fc{num} -> roi_heads.box_head.{stage}.fcs.{num-1}
            elif "roi_heads.box_head." in key and ".fc" in key:
                parts = key.split(".")
                # Find the fc number
                for i, part in enumerate(parts):
                    if part.startswith("fc") and part[2:].isdigit():
                        fc_num = int(part[2:])
                        parts[i] = f"fcs.{fc_num - 1}"
                        converted_key = ".".join(parts)
                        break
            
            converted[converted_key] = value
        
        return converted
