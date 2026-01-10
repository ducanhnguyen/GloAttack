# src/models/YOLOv5Model.py
import os
import sys

import numpy as np
import torch

# ✅ Setup YOLOv5 path trước khi import
yolov5_path = os.path.join(os.path.dirname(__file__), '../../yolov5')
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

from yolov5.utils.augmentations import letterbox

from src.BaseDetectionModel import BaseDetectionModel
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.loss import ComputeLoss


class YOLOv5Model(BaseDetectionModel):
    def __init__(self, device, model_path='yolov5s.pt'):
        super().__init__(device)
        self.model_path = model_path
        self.ratio = None
        self.pad = None
        self.load_model()

    def load_model(self):
        print("📦 Loading YOLOv5 model...")
        self.model = DetectMultiBackend(self.model_path, device=self.device)
        self.model.model.float().eval()
        self.names = self.model.names
        self.compute_loss_fn = ComputeLoss(self.model.model)
        print("✅ YOLOv5 model loaded successfully!")

    def preprocess_input(self, img_pil):
        img_np = np.array(img_pil)
        img_letterboxed, ratio, pad = letterbox(img_np, new_shape=640, auto=False)
        self.ratio = ratio  # Lưu để dùng cho targets
        self.pad = pad

        img_letterboxed = img_letterboxed.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img_letterboxed).unsqueeze(0).float().to(self.device) / 255.0
        return img_tensor

    def compute_loss(self, img_tensor, targets):
        self.model.model.train()
        pred = self.model.model(img_tensor)
        loss, *_ = self.compute_loss_fn(pred, targets)
        return loss

    def predict(self, img_tensor, conf_thresh=0.5):
        self.model.model.eval()
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=conf_thresh, iou_thres=0.45)

        boxes, scores, labels = [], [], []
        if pred[0] is not None:
            for det in pred[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                labels.append(int(cls))

        return boxes, scores, labels

    def prepare_targets(self, anns, img_pil, cat_id_to_index, ratio=None, pad=None):
        if ratio is None:
            ratio = self.ratio
        if pad is None:
            pad = self.pad

        targets = []
        gt_boxes = []
        gt_classes = []

        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in cat_id_to_index:
                continue

            cls = cat_id_to_index[cat_id]
            gt_classes.append(cls)

            # Scale box với letterbox
            gt_boxes.append(self.scale_box_with_letterbox(ann['bbox'], ratio, pad))

            # Tạo targets format YOLOv5
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / img_pil.width
            cy = (y + h / 2) / img_pil.height
            nw = w / img_pil.width
            nh = h / img_pil.height
            targets.append([0, cls, cx, cy, nw, nh])

        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        # Scale to image coordinates
        targets[:, 2] *= 640  # cx
        targets[:, 3] *= 640  # cy
        targets[:, 4] *= 640  # w
        targets[:, 5] *= 640  # h

        return targets, gt_boxes, gt_classes

    def scale_box_with_letterbox(self, box, ratio, pad):
        x, y, w, h = box
        x1 = x * ratio[0] + pad[0]
        y1 = y * ratio[1] + pad[1]
        x2 = (x + w) * ratio[0] + pad[0]
        y2 = (y + h) * ratio[1] + pad[1]
        return [x1, y1, x2, y2]
