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
DETR_RESNET50_DC5 = "detr-resnet-50-dc5" # facebook/detr-resnet-50-dc5

# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================
# False → chỉ chạy MODEL_TYPE. True → lần lượt RUNTIME_MODELS.
RUN_ALL_MODELS = True
MODEL_TYPE = YOLOV5L

NUM_IMAGE = 10
SAVE_DIR = "out"
IMAGE_LIST_DIR = "result/rq1/1k images"
RUNTIME_CSV = "result/rq1/runtime_until_success.csv"

ZERO_ATTACK = False
VISUALIZE_FREQUENCY = False
EXPORT_ADV_FOLDER = False
EXPORT_ORI_FOLDER = False

# Bật/tắt từng phương pháp (dùng trong if RUN_* ở main.py)
RUN_OURS = True
RUN_DCT_HIGH = True
RUN_DCT_MID = True
RUN_DCT_LOW = True
RUN_FGSM = True
RUN_NUMOD = True
RUN_PGD = True

GLOATTACK_CONFIGS = [
    {'epsilon': 0.09, 'max_iters': 1000, 'interval': 1},
]

DCT_EPS = 0.0627
DCT_ITERS = 50
FGSM_EPSILON = 3 / 255
PGD_EPSILON = 2 / 255
PGD_ITERS = 1000
PGD_ALPHA = 0.005
NUMOD_CONFIGS = [
    {'epsilon': 0.09, 'max_iters': 500, 'alpha': 0.5, 'lambda_sf': 0.5},
]

# Thứ tự hàng CSV (bỏ DETR ResNet-50-dc5)
RUNTIME_MODELS = [
    YOLOV5L,
    YOLOV5M,
    YOLOV5N,
    YOLOV5X,
    YOLOV5S,
    DETR_RESNET101,
    DETR_RESNET50,
    FASTER_RCNN_MOBILENET_320,
    FASTER_RCNN_RESNET50,
    FASTER_RCNN_RESNET50_V2,
]

MODEL_DISPLAY_NAMES = {
    YOLOV5L: "YOLOv5l",
    YOLOV5M: "YOLOv5m",
    YOLOV5N: "YOLOv5n",
    YOLOV5X: "YOLOv5x",
    YOLOV5S: "YOLOv5s",
    DETR_RESNET101: "DETR ResNet-101",
    DETR_RESNET50: "DETR ResNet-50",
    FASTER_RCNN_MOBILENET_320: "Faster R-CNN Mobilenet",
    FASTER_RCNN_RESNET50: "Faster R-CNN ResNet-50",
    FASTER_RCNN_RESNET50_V2: "Faster R-CNN ResNet-50v2",
}

# File danh sách ảnh: tên model.txt; Mobilenet dùng list RQ1 sẵn có.
IMAGE_LIST_ALIASES = {
    FASTER_RCNN_MOBILENET: "faster_rcnn_mobilenet_320.txt",
}

RUNTIME_CSV_COLUMNS = [
    "Model", "Ours", "DCT_high", "DCT_mid", "DCT_low", "FGSM", "NumbOD", "PGD",
]