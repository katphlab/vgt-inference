from __future__ import annotations

# Import to register backbone builder
import vgt_inference.ditod.VGTbackbone  # noqa: F401

from vgt_inference.modeling.backbone import BACKBONE_REGISTRY
from vgt_inference.modeling.meta_arch import GeneralizedRCNN
from vgt_inference.modeling.rpn import RPN, StandardRPNHead
from vgt_inference.modeling.anchors import DefaultAnchorGenerator
from vgt_inference.modeling.roi_heads import CascadeROIHeads, FastRCNNConvFCHead, FastRCNNOutputLayers
from vgt_inference.modeling.box_regression import Box2BoxTransform


def build_model(cfg):
    model_type = cfg.MODEL.META_ARCHITECTURE
    if model_type == "VGT":
        return build_vgt_model(cfg)
    raise ValueError(f"Unknown META_ARCHITECTURE: {model_type}")


def build_vgt_model(cfg):
    # Build backbone
    from vgt_inference.modeling.backbone import ShapeSpec
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, ShapeSpec())

    # Build RPN
    in_features = cfg.MODEL.RPN.IN_FEATURES
    # Get backbone output shapes for RPN head & anchor generator
    backbone_output_shape = backbone.output_shape()
    anchor_strides = [backbone_output_shape[f].stride for f in in_features]
    anchor_generator = DefaultAnchorGenerator(
        sizes=cfg.MODEL.ANCHOR_GENERATOR.SIZES,
        aspect_ratios=cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
        strides=anchor_strides,
        offset=cfg.MODEL.ANCHOR_GENERATOR.OFFSET if hasattr(cfg.MODEL.ANCHOR_GENERATOR, 'OFFSET') else 0.5,
    )

    in_channels = [backbone_output_shape[f].channels for f in in_features]
    # Use first channel for RPN head (they should all be same in FPN)
    rpn_head = StandardRPNHead(in_channels[0], anchor_generator.num_anchors_per_location()[0])
    
    box2box_transform_rpn = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
    proposal_generator = RPN(
        in_features=in_features,
        head=rpn_head,
        anchor_generator=anchor_generator,
        anchor_matcher=None,
        box2box_transform=box2box_transform_rpn,
        batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE if hasattr(cfg.MODEL.RPN, 'BATCH_SIZE_PER_IMAGE') else 256,
        positive_fraction=cfg.MODEL.RPN.POSITIVE_FRACTION if hasattr(cfg.MODEL.RPN, 'POSITIVE_FRACTION') else 0.5,
        pre_nms_topk=(cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST),
        post_nms_topk=(cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST),
        nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        min_box_size=cfg.MODEL.RPN.MIN_SIZE,
    )

    # Build ROI Heads
    roi_heads = build_roi_heads(cfg, backbone_output_shape)

    # Build model
    pixel_mean = cfg.MODEL.PIXEL_MEAN
    pixel_std = cfg.MODEL.PIXEL_STD
    
    # Import VGT class
    from vgt_inference.ditod.VGT import VGT
    
    model = VGT(
        backbone=backbone,
        proposal_generator=proposal_generator,
        roi_heads=roi_heads,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        input_format=cfg.INPUT.FORMAT,
        vocab_size=cfg.MODEL.WORDGRID.VOCAB_SIZE,
        hidden_size=cfg.MODEL.WORDGRID.HIDDEN_SIZE,
        embedding_dim=cfg.MODEL.WORDGRID.EMBEDDING_DIM,
        bros_embedding_path=cfg.MODEL.WORDGRID.MODEL_PATH,
        use_pretrain_weight=cfg.MODEL.WORDGRID.USE_PRETRAIN_WEIGHT,
        use_UNK_text=cfg.MODEL.WORDGRID.USE_UNK_TEXT,
    )
    
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    return model


def build_roi_heads(cfg, input_shape):
    # Build box head
    from vgt_inference.modeling.backbone import ShapeSpec
    pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
    # Create input shape with proper width/height for pooler output
    head_input_shape = ShapeSpec(
        channels=input_shape[cfg.MODEL.ROI_HEADS.IN_FEATURES[0]].channels,
        height=pooler_resolution,
        width=pooler_resolution,
    )
    box_head = FastRCNNConvFCHead(
        input_shape=head_input_shape,
        fc_dim=cfg.MODEL.ROI_BOX_HEAD.FC_DIM if hasattr(cfg.MODEL.ROI_BOX_HEAD, 'FC_DIM') else 1024,
        num_fc=cfg.MODEL.ROI_BOX_HEAD.NUM_FC,
    )
    
    # Build box predictor
    box_predictor = FastRCNNOutputLayers(
        input_size=box_head.output_shape.channels,
        num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        cls_agnostic_bbox_reg=cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
        test_score_thresh=cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        test_nms_thresh=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        test_topk_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
    )
    
    # Build pooler with explicit scales matching detectron2
    pooler_scales = tuple(1.0 / input_shape[k].stride for k in cfg.MODEL.ROI_HEADS.IN_FEATURES)

    return CascadeROIHeads(
        num_classes=cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        batch_size_per_image=cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        positive_fraction=cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
        proposal_append_gt=cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
        in_features=cfg.MODEL.ROI_HEADS.IN_FEATURES,
        pooler_resolution=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
        pooler_scales=pooler_scales,
        pooler_sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO if hasattr(cfg.MODEL.ROI_BOX_HEAD, 'POOLER_SAMPLING_RATIO') else 0,
        pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE if hasattr(cfg.MODEL.ROI_BOX_HEAD, 'POOLER_TYPE') else "ROIAlignV2",
        box_head=box_head,
        box_predictor=box_predictor,
    )
