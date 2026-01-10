from src.attack.myconfig import *
from src.models.FasterRCNNModel import FasterRCNNModel
from src.models.YOLOv5Model import YOLOv5Model
from src.models.DETRModel import DETRModel


MODEL_REGISTRY = {
    # ================= Faster R-CNN =================
    FASTER_RCNN_RESNET50: {
        "family": "faster_rcnn",
        "model_name": "fasterrcnn_resnet50_fpn",
    },
    FASTER_RCNN_RESNET50_V2: {
        "family": "faster_rcnn",
        "model_name": "fasterrcnn_resnet50_fpn_v2",
    },
    FASTER_RCNN_MOBILENET: {
        "family": "faster_rcnn",
        "model_name": "fasterrcnn_mobilenet_v3_large_fpn",
    },
    FASTER_RCNN_MOBILENET_320: {
        "family": "faster_rcnn",
        "model_name": "fasterrcnn_mobilenet_v3_large_320_fpn",
    },

    # ================= YOLOv5 =================
    YOLOV5N: {
        "family": "yolo",
        "model_path": "yolov5n.pt",
    },
    YOLOV5S: {
        "family": "yolo",
        "model_path": "yolov5s.pt",
    },
    YOLOV5M: {
        "family": "yolo",
        "model_path": "yolov5m.pt",
    },
    YOLOV5L: {
        "family": "yolo",
        "model_path": "yolov5l.pt",
    },
    YOLOV5X: {
        "family": "yolo",
        "model_path": "yolov5x.pt",
    },

    # ================= DETR =================
    DETR_RESNET50: {
        "family": "detr",
        "repo": "facebook/detr-resnet-50",
    },
    DETR_RESNET101: {
        "family": "detr",
        "repo": "facebook/detr-resnet-101",
    },
    DETR_RESNET50_DC5: {
        "family": "detr",
        "repo": "facebook/detr-resnet-50-dc5",
    },
}


def build_model(model_type, device):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MODEL_TYPE: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[model_type]

    if cfg["family"] == "faster_rcnn":
        return FasterRCNNModel(
            device,
            model_name=model_type
        )

    if cfg["family"] == "yolo":
        return YOLOv5Model(
            device,
            model_path=cfg["model_path"]
        )

    if cfg["family"] == "detr":
        return DETRModel(
            device,
            model_name=cfg["repo"]
        )

    raise RuntimeError(f"Invalid model family: {cfg['family']}")
