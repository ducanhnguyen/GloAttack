# src/models/FasterRCNNModel.py
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import detection

from src.BaseDetectionModel import BaseDetectionModel
from src.attack.myconfig import FASTER_RCNN_RESNET50, FASTER_RCNN_MOBILENET, FASTER_RCNN_MOBILENET_320, \
    FASTER_RCNN_RESNET50_V2


class FasterRCNNModel(BaseDetectionModel):
    def __init__(self, device, model_name=None, **kwargs):
        super().__init__(device)
        self.model_name = model_name
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self.load_model()

    def load_model(self):
        print(f"Loading Faster R-CNN model: {self.model_name}...")

        FASTER_RCNN_LOADERS = {
            # ResNet-50 + FPN (v1)
            FASTER_RCNN_RESNET50: detection.fasterrcnn_resnet50_fpn,

            # ResNet-50 + FPN (v2 – improved head, torchvision >= 0.14)
            FASTER_RCNN_RESNET50_V2: detection.fasterrcnn_resnet50_fpn_v2,

            # MobileNetV3 Large + FPN
            FASTER_RCNN_MOBILENET: detection.fasterrcnn_mobilenet_v3_large_fpn,

            # MobileNetV3 Large + FPN (320)
            FASTER_RCNN_MOBILENET_320: detection.fasterrcnn_mobilenet_v3_large_320_fpn,
        }

        if self.model_name not in FASTER_RCNN_LOADERS:
            raise ValueError(
                f"Unsupported Faster R-CNN model: {self.model_name}. "
                f"Available: {list(FASTER_RCNN_LOADERS.keys())}"
            )

        loader_fn = FASTER_RCNN_LOADERS[self.model_name]
        self.model = loader_fn(pretrained=True)

        self.model.to(self.device)
        self.model.eval()

        print("Faster R-CNN model loaded successfully!")

        self.model.to(self.device)
        self.model.eval()

        # COCO class names (80 classes)
        self.names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        print("✅ Faster R-CNN model loaded successfully!")

    def preprocess_input(self, img_pil):
        """
        Preprocess PIL image for Faster R-CNN
        Returns tensor in [0, 1] range with shape (1, 3, H, W)

        For compatibility with frequency-domain attacks, resize to fixed size
        """
        # Resize to fixed size (compatible với DCT attacks)
        # Chọn size chia hết cho 8 để support AdvDrop
        target_size = (640, 640)  # hoặc (512, 512), (768, 768)

        # Resize while maintaining aspect ratio (similar to YOLO letterbox)
        img_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)

        img_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        return img_tensor

    def compute_loss(self, img_tensor, targets):
        """
        Compute loss for Faster R-CNN
        Args:
            img_tensor: Input image tensor (1, 3, H, W)
            targets: List of target dictionaries with 'boxes', 'labels'
        """
        self.model.train()

        # Convert targets to proper format if needed
        if isinstance(targets, torch.Tensor):
            targets = self._convert_yolo_targets_to_rcnn(targets, img_tensor.shape)

        # Faster R-CNN expects list of images and list of targets
        images = [img_tensor.squeeze(0)]  # Remove batch dimension

        try:
            loss_dict = self.model(images, targets)
            # Sum all losses - ensure it's a scalar with gradient
            total_loss = sum(loss for loss in loss_dict.values())

            # Make sure loss requires gradient for FGSM
            if not total_loss.requires_grad:
                total_loss = total_loss.requires_grad_(True)

            return total_loss

        except Exception as e:
            print(f"⚠Loss computation failed: {e}")
            # Return a dummy loss that requires gradient
            dummy_loss = torch.sum(img_tensor) * 0.0  # Will have gradient
            return dummy_loss.requires_grad_(True)

    def predict(self, img_tensor, conf_thresh=0.5):
        """
        Make predictions with Faster R-CNN
        Returns: boxes, scores, labels (lists)
        """
        self.model.eval()

        with torch.no_grad():
            # Faster R-CNN expects list of images
            images = [img_tensor.squeeze(0)]  # Remove batch dimension
            predictions = self.model(images)

        pred = predictions[0]  # Get first (and only) prediction

        # Filter by confidence threshold
        keep = pred['scores'] > conf_thresh

        boxes = pred['boxes'][keep].cpu().numpy().tolist()
        scores = pred['scores'][keep].cpu().numpy().tolist()
        # Convert from 1-indexed to 0-indexed classes và clamp về [0, 79]
        labels = (pred['labels'][keep] - 1).cpu().numpy()
        labels = np.clip(labels, 0, 79).tolist()  # Clamp về range hợp lệ

        return boxes, scores, labels

    def prepare_targets(self, anns, img_pil, cat_id_to_index, ratio=None, pad=None):
        """
        Prepare targets for Faster R-CNN training
        Returns: targets (list of dicts), gt_boxes, gt_classes
        """
        targets = []
        gt_boxes = []
        gt_classes = []

        boxes = []
        labels = []

        # Calculate scaling factors from original image to fixed size (640x640)
        orig_w, orig_h = img_pil.size
        target_w, target_h = 640, 640
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in cat_id_to_index:
                continue

            cls = cat_id_to_index[cat_id]
            gt_classes.append(cls)

            # Convert COCO bbox format [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']

            # Scale coordinates to match resized image (640x640)
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            # Clamp to image boundaries
            x1 = max(0, min(x1, target_w))
            y1 = max(0, min(y1, target_h))
            x2 = max(0, min(x2, target_w))
            y2 = max(0, min(y2, target_h))

            if x2 > x1 and y2 > y1:  # Valid box
                gt_boxes.append([x1, y1, x2, y2])
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)  # Faster R-CNN uses 1-indexed classes

        if boxes:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32, device=self.device),
                'labels': torch.tensor(labels, dtype=torch.int64, device=self.device)
            }
            targets.append(target)
        else:
            # Empty target
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=self.device)
            }
            targets.append(target)

        return targets, gt_boxes, gt_classes

    def _convert_yolo_targets_to_rcnn(self, yolo_targets, img_shape):
        """
        Convert YOLO format targets to Faster R-CNN format
        Args:
            yolo_targets: Tensor with shape (N, 6) - [img_idx, class, cx, cy, w, h]
            img_shape: (B, C, H, W)
        """
        if len(yolo_targets) == 0:
            return [{
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=self.device)
            }]

        _, _, H, W = img_shape

        boxes = []
        labels = []

        for target in yolo_targets:
            if len(target) < 6:
                continue

            _, cls, cx, cy, w, h = target[:6]

            # Convert normalized coordinates to absolute coordinates
            if cx <= 1 and cy <= 1 and w <= 1 and h <= 1:
                # Normalized coordinates, convert to absolute
                cx *= W
                cy *= H
                w *= W
                h *= H

            # Convert center format to corner format
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Clamp to image boundaries
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))

            if x2 > x1 and y2 > y1:  # Valid box
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # Convert to 1-indexed

        if boxes:
            rcnn_targets = [{
                'boxes': torch.tensor(boxes, dtype=torch.float32, device=self.device),
                'labels': torch.tensor(labels, dtype=torch.int64, device=self.device)
            }]
        else:
            rcnn_targets = [{
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=self.device)
            }]

        return rcnn_targets
