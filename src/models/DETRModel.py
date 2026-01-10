# src/models/DETRModel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

from src.BaseDetectionModel import BaseDetectionModel


class DETRModel(BaseDetectionModel):
    def __init__(self, device, model_name=None, cat_id_to_index=None, **kwargs):
        super().__init__(device)
        self.model_name = model_name
        self.processor = None
        self.load_model()
        self.cat_id_to_index = cat_id_to_index or {}

    def _create_coco_to_detr_mapping(self):

        coco_cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                        80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

        mapping = {}
        for detr_idx, coco_id in enumerate(coco_cat_ids):
            mapping[coco_id] = detr_idx

        return mapping

    def compute_loss(self, img_tensor, targets):

        self.model.train()

        # Re-normalize cho DETR
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        normalized_tensor = (img_tensor - mean) / std

        try:
            # DETR expects specific format for labels
            if isinstance(targets, list) and len(targets) > 0:
                target = targets[0]  # Take first (and only) image target

                if isinstance(target, dict) and 'boxes' in target and 'labels' in target:
                    # Convert boxes to normalized [cx, cy, w, h] format
                    boxes = target['boxes']  # [x1, y1, x2, y2]
                    labels = target['labels']  # COCO category IDs

                    H, W = img_tensor.shape[-2:]

                    # Convert [x1, y1, x2, y2] to normalized [cx, cy, w, h]
                    if len(boxes) > 0:
                        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                        cx = (x1 + x2) / 2 / W
                        cy = (y1 + y2) / 2 / H
                        w = (x2 - x1) / W
                        h = (y2 - y1) / H

                        normalized_boxes = torch.stack([cx, cy, w, h], dim=1)

                        #CRITICAL: DETR expects 'class_labels' not 'labels'
                        formatted_labels = [{
                            'class_labels': labels,
                            'boxes': normalized_boxes
                        }]
                    else:
                        # Empty targets
                        formatted_labels = [{
                            'class_labels': torch.zeros((0,), dtype=torch.int64, device=self.device),
                            'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                        }]

                    # print(f"🔍 DETR loss input: {len(formatted_labels[0]['class_labels'])} targets")

                    # Forward pass with correct format
                    outputs = self.model(pixel_values=normalized_tensor, labels=formatted_labels)
                    loss = outputs.loss

                    if not loss.requires_grad:
                        loss = loss.requires_grad_(True)

                    return loss

                else:
                    print("Invalid target format")
                    dummy_loss = torch.sum(normalized_tensor) * 0.0
                    return dummy_loss.requires_grad_(True)

            else:
                print("Empty or invalid targets")
                dummy_loss = torch.sum(normalized_tensor) * 0.0
                return dummy_loss.requires_grad_(True)

        except Exception as e:
            print(f"DETR loss computation failed: {e}")
            # Return gradient-enabled dummy loss
            dummy_loss = torch.sum(normalized_tensor) * 0.0
            return dummy_loss.requires_grad_(True)


    def get_coco_id_to_index_mapping(self):
        # COCO class IDs (80 classes, missing some numbers)
        coco_class_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

        # Map to sequential indices 0-79
        return {coco_id: idx for idx, coco_id in enumerate(coco_class_ids)}


    def preprocess_input(self, img_pil):

        target_size = (800, 800)  # Changed from (640, 640)
        img_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)

        transform = T.Compose([
            T.ToTensor(),  # Converts to [0, 1] and changes to (C, H, W)
        ])

        img_tensor = transform(img_resized).unsqueeze(0).to(self.device)
        return img_tensor

    def predict(self, img_tensor, conf_thresh=0.5):

        self.model.eval()

        # Re-normalize cho DETR
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        normalized_tensor = (img_tensor - mean) / std

        with torch.no_grad():
            outputs = self.model(normalized_tensor)


        H, W = img_tensor.shape[-2:]
        target_sizes = torch.tensor([[H, W]]).to(self.device)

        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_thresh
        )[0]

        if len(results["scores"]) == 0:
            print("⚠️ No detections above threshold")
            return [], [], []


        boxes = results["boxes"].cpu().numpy().tolist()  # [x1, y1, x2, y2] format
        scores = results["scores"].cpu().numpy().tolist()
        coco_labels = results["labels"].cpu().numpy().tolist()  # COCO category IDs

        # Map COCO category IDs to sequential indices (0-79)
        final_labels = []
        for coco_id in coco_labels:
            if self.cat_id_to_index and coco_id in self.cat_id_to_index:
                sequential_idx = self.cat_id_to_index[coco_id]
                final_labels.append(sequential_idx)
            else:
                # print(f"⚠️ COCO ID {coco_id} not found in mapping")
                final_labels.append(0)  # fallback

        return boxes, scores, final_labels

    def load_model(self):
        print(f"📦 Loading DETR model: {self.model_name}...")

        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model = DetrForObjectDetection.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()

        self.names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        print("✅ DETR model loaded successfully!")

    def prepare_targets(self, anns, img_pil, cat_id_to_index, ratio=None, pad=None):
        """
        ✅ Prepare targets với COCO category IDs
        """
        targets = []
        gt_boxes = []
        gt_classes = []

        boxes = []
        labels = []

        # Scale to 800x800 (chuẩn DETR)
        orig_w, orig_h = img_pil.size
        target_w, target_h = 800, 800
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        print(f"🔍 Image resize: {orig_w}x{orig_h} -> {target_w}x{target_h}")

        for ann in anns:
            cat_id = ann['category_id']  # COCO category ID (1-90)

            if cat_id not in cat_id_to_index:
                continue

            # Sequential index for evaluation
            sequential_idx = cat_id_to_index[cat_id]
            gt_classes.append(sequential_idx)

            # Convert COCO bbox [x, y, w, h] to scaled [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']

            # Scale coordinates
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            # Clamp to boundaries
            x1 = max(0, min(x1, target_w))
            y1 = max(0, min(y1, target_h))
            x2 = max(0, min(x2, target_w))
            y2 = max(0, min(y2, target_h))

            if x2 > x1 and y2 > y1:  # Valid box
                gt_boxes.append([x1, y1, x2, y2])
                boxes.append([x1, y1, x2, y2])


                labels.append(cat_id)  # Giữ nguyên COCO category ID

        if boxes:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32, device=self.device),
                'labels': torch.tensor(labels, dtype=torch.int64, device=self.device)
            }
            targets.append(target)
            print(f"🔍 Prepared {len(boxes)} targets with COCO IDs: {labels[:3]}")
        else:
            # Empty target
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=self.device)
            }
            targets.append(target)
            print("🔍 No valid targets")

        return targets, gt_boxes, gt_classes

    def _format_targets_for_detr(self, targets, img_shape):
        """Format targets for DETR loss computation"""
        if not targets or len(targets) == 0:
            return []

        formatted_targets = []
        for target in targets:
            if target['boxes'].shape[0] == 0:
                continue

            # DETR expects targets as a list of dicts với specific format
            formatted_target = {
                'boxes': target['boxes'],  # Already in normalized [cx, cy, w, h] format
                'labels': target['labels']
            }
            formatted_targets.append(formatted_target)

        return formatted_targets

    def _convert_detr_boxes_to_absolute(self, boxes, img_w, img_h):
        """
        Convert DETR normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
        """
        if len(boxes) == 0:
            return torch.empty((0, 4))

        cx, cy, w, h = boxes.unbind(1)

        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        return torch.stack([x1, y1, x2, y2], dim=1)

    def _convert_yolo_targets_to_detr(self, yolo_targets, img_shape):
        """
        Convert YOLO format targets to DETR format
        Args:
            yolo_targets: Tensor với shape (N, 6) - [img_idx, class, cx, cy, w, h]
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

            # YOLO targets are already normalized, but DETR expects [cx, cy, w, h]
            # Check if coordinates are normalized
            if cx <= 1 and cy <= 1 and w <= 1 and h <= 1:
                # Already normalized
                boxes.append([cx, cy, w, h])
            else:
                # Convert to normalized
                cx_norm = cx / W
                cy_norm = cy / H
                w_norm = w / W
                h_norm = h / H
                boxes.append([cx_norm, cy_norm, w_norm, h_norm])

            # Map YOLO class to DETR class if needed
            detr_cls = int(cls)
            labels.append(detr_cls)

        if boxes:
            detr_targets = [{
                'boxes': torch.tensor(boxes, dtype=torch.float32, device=self.device),
                'labels': torch.tensor(labels, dtype=torch.int64, device=self.device)
            }]
        else:
            detr_targets = [{
                'boxes': torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=self.device)
            }]

        return detr_targets


class DETRLoss(nn.Module):
    """
    Custom DETR loss implementation for better control
    """

    def __init__(self, num_classes, weight_dict=None):
        super().__init__()
        self.num_classes = num_classes

        if weight_dict is None:
            self.weight_dict = {
                'loss_ce': 1,
                'loss_bbox': 5,
                'loss_giou': 2
            }
        else:
            self.weight_dict = weight_dict

    def forward(self, outputs, targets):
        """
        Compute DETR loss với Hungarian matching
        """
        # Get predictions
        pred_logits = outputs['pred_logits']  # (batch_size, num_queries, num_classes+1)
        pred_boxes = outputs['pred_boxes']  # (batch_size, num_queries, 4)

        # Perform Hungarian matching
        indices = self.hungarian_matching(pred_logits, pred_boxes, targets)

        # Compute classification loss
        loss_ce = self.compute_classification_loss(pred_logits, targets, indices)

        # Compute bbox regression loss
        loss_bbox = self.compute_bbox_loss(pred_boxes, targets, indices)

        # Compute GIoU loss
        loss_giou = self.compute_giou_loss(pred_boxes, targets, indices)

        # Weighted sum
        total_loss = (self.weight_dict['loss_ce'] * loss_ce +
                      self.weight_dict['loss_bbox'] * loss_bbox +
                      self.weight_dict['loss_giou'] * loss_giou)

        return total_loss

    def hungarian_matching(self, pred_logits, pred_boxes, targets):
        """Simplified Hungarian matching"""
        # This is a simplified version
        # In practice, you'd want to implement full Hungarian algorithm
        indices = []

        for batch_idx, target in enumerate(targets):
            if len(target['labels']) == 0:
                indices.append((torch.empty(0, dtype=torch.int64),
                                torch.empty(0, dtype=torch.int64)))
                continue

            # Simple matching - take top predictions
            num_gt = len(target['labels'])
            pred_idx = torch.arange(num_gt)
            tgt_idx = torch.arange(num_gt)

            indices.append((pred_idx, tgt_idx))

        return indices

    def compute_classification_loss(self, pred_logits, targets, indices):
        """Compute classification loss"""
        batch_size, num_queries = pred_logits.shape[:2]

        # Create target classes
        target_classes = torch.full((batch_size, num_queries), self.num_classes,
                                    dtype=torch.int64, device=pred_logits.device)

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[batch_idx, pred_idx] = targets[batch_idx]['labels'][tgt_idx]

        # Compute cross entropy loss
        loss_ce = F.cross_entropy(pred_logits.flatten(0, 1), target_classes.flatten())

        return loss_ce

    def compute_bbox_loss(self, pred_boxes, targets, indices):
        """Compute L1 bbox loss"""
        losses = []

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            pred_bbox = pred_boxes[batch_idx, pred_idx]
            tgt_bbox = targets[batch_idx]['boxes'][tgt_idx]

            loss = F.l1_loss(pred_bbox, tgt_bbox, reduction='mean')
            losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)

    def compute_giou_loss(self, pred_boxes, targets, indices):
        """Compute GIoU loss"""
        # Simplified GIoU implementation
        losses = []

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            # For simplicity, use L1 loss instead of actual GIoU
            pred_bbox = pred_boxes[batch_idx, pred_idx]
            tgt_bbox = targets[batch_idx]['boxes'][tgt_idx]

            loss = F.l1_loss(pred_bbox, tgt_bbox, reduction='mean')
            losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)