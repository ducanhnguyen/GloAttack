# src/attack/myconfig.py
# ============================================================================
# 🔧 MODEL
# ============================================================================
# ================= Faster R-CNN =================
FASTER_RCNN_RESNET50 = "faster_rcnn_resnet50" #https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
FASTER_RCNN_MOBILENET = "faster_rcnn_mobilenet" #https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"
FASTER_RCNN_MOBILENET_320 = "faster_rcnn_mobilenet_320" #https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth

FASTER_RCNN_RESNET50_V2 = "faster_rcnn_resnet50_v2" # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth

# ================= YOLOv5 =================
YOLOV5S = "yolov5s" #https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
YOLOV5M = "yolov5m" #https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
YOLOV5L = "yolov5l" #https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
YOLOV5X = "yolov5x" # https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
YOLOV5N = "yolov5n" # https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt

# ================= DETR =================
DETR_RESNET50 = "detr-resnet-50" #facebook/detr-resnet-50
DETR_RESNET101 = "detr-resnet-101" #facebook/detr-resnet-101
DETR_RESNET50_DC5 = "detr-resnet-50-dc5" # acebook/detr-resnet-50-dc5

# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================
MODEL_TYPE = YOLOV5L

NUM_IMAGE = 1
SAVE_DIR = "out"

ZERO_ATTACK = False
VISUALIZE_FREQUENCY=False
EXPORT_ADV_FOLDER=False
EXPORT_ORI_FOLDER=False

ADVLOGO_CONFIGS = [
            # {'epsilon': 0.01, 'max_iters': 500, 'interval': 1}
            # ,
            # {'epsilon': 0.03, 'max_iters': 500, 'interval': 1},
            # {'epsilon': 0.05, 'max_iters': 500, 'interval': 1},
            # {'epsilon': 0.07, 'max_iters': 500, 'interval': 1},
            {'epsilon': 0.09, 'max_iters': 1000, 'interval': 1},
        ]