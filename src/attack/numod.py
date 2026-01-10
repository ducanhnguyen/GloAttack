# https://arxiv.org/pdf/2412.16955
import torch
import numpy as np
from scipy.fftpack import dct, idct
from src.myutils import (
    load_image_and_targets, exportImage, compute_map_per_image
)


def numod(model, sample_img_ids, epsilon=0.1, max_iters=50, alpha=0.5, lambda_sf=0.5,
          k_reg=5, k_cls=5, csv_writer=None, save_dir="out"):
    """
    NumbOD Attack Implementation theo paper chính thức

    Args:
        model: BaseDetectionModel instance
        sample_img_ids: List of image IDs
        epsilon: Maximum perturbation bound
        max_iters: Maximum iterations
        alpha: Weight for frequency components
        lambda_sf: Weight for spatial-frequency fusion
        k_reg: Top-k boxes for regression track
        k_cls: Top-k boxes for classification track
        csv_writer: CSV writer for logging
        save_dir: Directory to save results
    """
    success = 0
    print(f"🎯 Starting NumbOD attack with epsilon={epsilon}, max_iters={max_iters}")

    total_images = len(sample_img_ids)
    for idx, img_id in enumerate(sample_img_ids, 1):
        print("=" * 50)
        print(f"🖼️ Image {img_id} [{idx:2d}/{total_images}]")

        try:
            # === LOAD IMAGE VÀ TARGETS ===
            result = load_image_and_targets(img_id, model)
            if result is None:
                print(f"⚠️ Skipping image {img_id}: No valid annotations")
                continue
            img_tensor, ori_tensor, gt_boxes, gt_classes, targets = result

            # === Baseline evaluation ===
            pred_boxes_orig, pred_scores_orig, pred_labels_orig = model.predict(ori_tensor)
            map_orig = compute_map_per_image(
                pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                gt_boxes, gt_classes
            )

            # === NumbOD Attack ===
            # Initialize perturbation
            delta = torch.zeros_like(ori_tensor, requires_grad=True, device=ori_tensor.device)
            optimizer = torch.optim.Adam([delta], lr=0.01)

            best_adv = None
            best_map_adv = float('inf')
            attack_successful = False
            final_iteration = 0

            for step in range(max_iters):
                optimizer.zero_grad()

                # Create adversarial example
                adv_tensor = torch.clamp(ori_tensor + delta, 0, 1)

                # === GET CURRENT PREDICTIONS FOR TARGET SELECTION ===
                pred_boxes_current, pred_scores_current, pred_labels_current = model.predict(adv_tensor)

                # === DUAL-TRACK TARGET SELECTION ===
                selected_targets = dual_track_target_selection(
                    pred_boxes_current, pred_scores_current, pred_labels_current,
                    targets, gt_boxes, gt_classes, k_reg=k_reg, k_cls=k_cls
                )

                # === SPATIAL COORDINATED DEVIATION ATTACK ===
                spatial_loss = model.compute_loss(adv_tensor, selected_targets)
                if spatial_loss is None or not isinstance(spatial_loss, torch.Tensor):
                    spatial_loss = torch.tensor(0.0, requires_grad=True, device=ori_tensor.device)
                else:
                    spatial_loss = -spatial_loss  # Negative to minimize detection

                # === SPATIAL-FREQUENCY FUSION ATTACK ===
                freq_fusion_loss = spatial_frequency_fusion_attack(ori_tensor, adv_tensor, lambda_sf)

                # === CRITICAL FREQUENCY INTERFERENCE ===
                freq_interference_loss = critical_frequency_interference_attack(ori_tensor, adv_tensor)

                # === TOTAL LOSS COMBINATION (Equation 1) ===
                total_loss = spatial_loss + alpha * (freq_fusion_loss + freq_interference_loss)

                if total_loss.dim() > 0:
                    total_loss = total_loss.mean()

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Project perturbation to epsilon ball (L∞ constraint)
                with torch.no_grad():
                    delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                    delta.data = torch.clamp(ori_tensor + delta.data, 0, 1) - ori_tensor

                # === CHECK SUCCESS AFTER EACH ITERATION ===
                pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_tensor)
                map_adv = compute_map_per_image(
                    pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                    gt_boxes, gt_classes
                )

                print(f"📊 Iter {step + 1}/{max_iters}: Spatial={spatial_loss.item():.4f}, "
                      f"FreqFusion={freq_fusion_loss.item():.4f}, "
                      f"FreqInterf={freq_interference_loss.item():.4f}, "
                      f"Total={total_loss.item():.4f}, mAP: {map_orig:.3f}→{map_adv:.3f}")

                # Save best adversarial example
                if map_adv < best_map_adv:
                    best_map_adv = map_adv
                    best_adv = adv_tensor.clone()

                # Check if attack successful
                if map_adv < map_orig:
                    attack_successful = True
                    final_iteration = step + 1
                    print(f"✅ Attack successful at iteration {final_iteration}!")
                    break

            # === FINAL EVALUATION AND EXPORT ===
            if attack_successful and best_adv is not None:
                # Get final predictions for export
                pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(best_adv)
                map_adv = compute_map_per_image(
                    pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                    gt_boxes, gt_classes
                )

                variant_name = f"numod_eps{epsilon}_k{k_reg}_{k_cls}_iter{final_iteration}"
                exportImage(ori_tensor, gt_classes, best_adv, model,
                            gt_boxes, img_id, save_dir, map_orig, map_adv,
                            pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                            pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                            variant_name=variant_name, csv_writer=csv_writer)
                success += 1
                print(f"✅ Attack successful on image {img_id} after {final_iteration} iterations")
            else:
                print(f"❌ Attack failed on image {img_id} after {max_iters} iterations")

        except Exception as e:
            print(f"❌ Error attacking image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # === THỐNG KÊ KẾT QUẢ ===
    success_rate = (success / len(sample_img_ids)) * 100 if len(sample_img_ids) > 0 else 0
    print(f"📊 NumbOD {model.__class__.__name__}: "
          f"Successful attacks: {success}/{len(sample_img_ids)} ({success_rate:.2f}%)")
    return success


def dual_track_target_selection(pred_boxes, pred_scores, pred_labels,
                                targets, gt_boxes, gt_classes, k_reg=5, k_cls=5):
    """
    Dual-track Target Selection Strategy theo Figure 2 trong paper
    Compatible with both YOLO tensor format and Faster R-CNN dict format

    Args:
        pred_boxes: Current predicted boxes
        pred_scores: Current prediction confidence scores
        pred_labels: Current predicted labels
        targets: Original targets (tensor for YOLO, list of dicts for Faster R-CNN)
        gt_boxes: Ground truth boxes
        gt_classes: Ground truth classes
        k_reg: Top-k boxes for regression track
        k_cls: Top-k boxes for classification track

    Returns:
        selected_targets: Selected targets in same format as input
    """
    if not pred_boxes or len(pred_boxes) == 0:
        return targets

    device = targets[0]['boxes'].device if isinstance(targets, list) else targets.device

    # === REGRESSION TRACK: Select top-k highest IoU boxes ===
    reg_indices = []
    if pred_boxes:
        ious = []
        for i, pred_box in enumerate(pred_boxes):
            max_iou = 0
            for gt_box in gt_boxes:
                iou = compute_iou_boxes(pred_box, gt_box)
                max_iou = max(max_iou, iou)
            ious.append(max_iou)

        # Select top-k regression targets
        if ious:
            k_reg_actual = min(k_reg, len(ious))
            reg_indices = torch.topk(torch.tensor(ious), k_reg_actual)[1].tolist()

    # === CLASSIFICATION TRACK: Select top-k highest confidence boxes ===
    cls_indices = []
    if pred_scores:
        k_cls_actual = min(k_cls, len(pred_scores))
        cls_indices = torch.topk(torch.tensor(pred_scores), k_cls_actual)[1].tolist()

    # === COMBINE SELECTED INDICES ===
    all_selected_indices = list(set(reg_indices + cls_indices))

    # === HANDLE DIFFERENT TARGET FORMATS ===
    if len(all_selected_indices) > 0:
        if isinstance(targets, list) and len(targets) > 0 and isinstance(targets[0], dict):
            # Faster R-CNN format: list of dicts
            return _select_faster_rcnn_targets(targets, all_selected_indices, pred_boxes, pred_labels, gt_boxes,
                                               gt_classes)
        elif isinstance(targets, torch.Tensor):
            # YOLO format: tensor
            return _select_yolo_targets(targets, all_selected_indices, pred_boxes, pred_labels, gt_boxes, gt_classes)

    return targets


def _select_faster_rcnn_targets(targets, selected_indices, pred_boxes, pred_labels, gt_boxes, gt_classes):
    """
    Select targets for Faster R-CNN format (list of dicts)
    """
    if not targets or len(targets) == 0:
        return targets

    original_target = targets[0]  # Faster R-CNN has one target dict per image
    device = original_target['boxes'].device

    # Create new target with selected boxes
    selected_boxes = []
    selected_labels = []

    for idx in selected_indices:
        if idx < len(pred_boxes):
            # Find best matching ground truth for this prediction
            pred_box = pred_boxes[idx]
            pred_label = pred_labels[idx] if idx < len(pred_labels) else 0

            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = 0
            for j, gt_box in enumerate(gt_boxes):
                iou = compute_iou_boxes(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_gt_idx < len(gt_boxes) and best_gt_idx < len(gt_classes):
                selected_boxes.append(gt_boxes[best_gt_idx])
                selected_labels.append(gt_classes[best_gt_idx] + 1)  # Faster R-CNN uses 1-indexed

    if selected_boxes:
        new_target = {
            'boxes': torch.tensor(selected_boxes, dtype=torch.float32, device=device),
            'labels': torch.tensor(selected_labels, dtype=torch.int64, device=device)
        }
        return [new_target]

    return targets


def _select_yolo_targets(targets, selected_indices, pred_boxes, pred_labels, gt_boxes, gt_classes):
    """
    Select targets for YOLO format (tensor)
    """
    if len(targets) == 0:
        return targets

    device = targets.device
    selected_targets = []

    for idx in selected_indices:
        if idx < len(pred_boxes):
            # Find best matching ground truth target for this prediction
            pred_box = pred_boxes[idx]
            pred_label = pred_labels[idx] if idx < len(pred_labels) else 0

            # Find best matching ground truth
            best_iou = 0
            best_target_idx = 0
            for j, gt_box in enumerate(gt_boxes):
                iou = compute_iou_boxes(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j

            if best_target_idx < len(targets):
                selected_targets.append(targets[best_target_idx])

    if selected_targets:
        return torch.stack(selected_targets)

    return targets

def compute_iou_boxes(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def spatial_frequency_fusion_attack(ori_tensor, adv_tensor, lambda_sf=0.5):
    """
    Spatial-Frequency Fusion Attack theo Equation (2): J_fs = J_lfc - λ_sf * J_hfc
    Sử dụng DCT để tách low/high frequency components
    """
    device = ori_tensor.device

    # Convert to numpy for DCT processing
    ori_np = ori_tensor.squeeze().detach().cpu().numpy()  # (C, H, W)
    adv_np = adv_tensor.squeeze().detach().cpu().numpy()  # (C, H, W)

    total_lfc_loss = 0.0  # Low-frequency component loss
    total_hfc_loss = 0.0  # High-frequency component loss

    for c in range(ori_np.shape[0]):  # For each channel
        ori_channel = ori_np[c]  # (H, W)
        adv_channel = adv_np[c]  # (H, W)

        # Apply 2D DCT
        ori_dct = dct(dct(ori_channel.T, norm='ortho').T, norm='ortho')
        adv_dct = dct(dct(adv_channel.T, norm='ortho').T, norm='ortho')

        H, W = ori_dct.shape

        # Define low-frequency region (top-left quarter)
        lf_h, lf_w = H // 4, W // 4

        # Low-frequency components (LFC) - top-left region
        ori_lfc = ori_dct[:lf_h, :lf_w]
        adv_lfc = adv_dct[:lf_h, :lf_w]
        lfc_loss = np.mean((ori_lfc - adv_lfc) ** 2)
        total_lfc_loss += lfc_loss

        # High-frequency components (HFC) - everything else
        ori_hfc = ori_dct.copy()
        adv_hfc = adv_dct.copy()
        ori_hfc[:lf_h, :lf_w] = 0  # Zero out low-frequency
        adv_hfc[:lf_h, :lf_w] = 0
        hfc_loss = np.mean((ori_hfc - adv_hfc) ** 2)
        total_hfc_loss += hfc_loss

    # Spatial-Frequency Fusion: J_fs = J_lfc - λ_sf * J_hfc
    # We want to preserve low-freq (minimize lfc_loss) and enhance high-freq changes
    fusion_loss = total_lfc_loss - lambda_sf * total_hfc_loss

    return torch.tensor(fusion_loss, dtype=torch.float32, device=device, requires_grad=True)


def critical_frequency_interference_attack(ori_tensor, adv_tensor):
    """
    Critical Frequency Interference Attack theo Section 3.3
    Focus on disrupting critical frequency components
    """
    device = ori_tensor.device

    # Convert to numpy for frequency domain processing
    ori_np = ori_tensor.squeeze().detach().cpu().numpy()  # (C, H, W)
    adv_np = adv_tensor.squeeze().detach().cpu().numpy()  # (C, H, W)

    total_interference_loss = 0.0

    for c in range(ori_np.shape[0]):  # For each channel
        ori_channel = ori_np[c]
        adv_channel = adv_np[c]

        # Apply 2D DCT
        ori_dct = dct(dct(ori_channel.T, norm='ortho').T, norm='ortho')
        adv_dct = dct(dct(adv_channel.T, norm='ortho').T, norm='ortho')

        H, W = ori_dct.shape

        # Create frequency interference mask
        # Focus on mid-frequency components (critical for object detection)
        freq_mask = create_critical_frequency_mask(H, W)

        # Apply mask to focus on critical frequencies
        ori_critical = ori_dct * freq_mask
        adv_critical = adv_dct * freq_mask

        # Compute interference loss - encourage changes in critical frequencies
        interference_loss = -np.mean((ori_critical - adv_critical) ** 2)  # Negative to encourage changes
        total_interference_loss += interference_loss

    return torch.tensor(total_interference_loss, dtype=torch.float32, device=device, requires_grad=True)


def create_critical_frequency_mask(H, W, low_ratio=0.2, high_ratio=0.8):
    """
    Create mask for critical frequency components
    Focus on mid-frequency range which is important for object detection
    """
    # Create radial frequency mask
    center_h, center_w = H // 2, W // 2
    y, x = np.ogrid[:H, :W]

    # Distance from DC component (center)
    distance = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
    max_distance = np.sqrt(center_h ** 2 + center_w ** 2)

    # Normalize distance
    norm_distance = distance / max_distance

    # Create band-pass mask for critical frequencies
    mask = np.zeros((H, W))
    mask[(norm_distance >= low_ratio) & (norm_distance <= high_ratio)] = 1.0

    return mask


def background_separation_mask(img_tensor, targets, threshold=0.3):
    """
    Tạo mask để tập trung perturbation vào object regions
    Giữ nguyên implementation cũ vì đã đúng concept
    """
    device = img_tensor.device
    _, _, h, w = img_tensor.shape
    mask = torch.zeros((h, w), device=device)

    # Create object regions from targets
    for target in targets:
        if len(target) >= 6:
            _, cls, cx, cy, width, height = target[:6]
        else:
            continue

        # Convert normalized coordinates to pixel coordinates
        x1 = int(max(0, (cx - width / 2) * w))
        y1 = int(max(0, (cy - height / 2) * h))
        x2 = int(min(w, (cx + width / 2) * w))
        y2 = int(min(h, (cy + height / 2) * h))

        # Expand region slightly for better coverage
        expand = 20
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(w, x2 + expand)
        y2 = min(h, y2 + expand)

        mask[y1:y2, x1:x2] = 1.0

    # Add some background with lower weight
    background_mask = (mask == 0).float() * threshold
    final_mask = mask + background_mask

    return final_mask.unsqueeze(0).unsqueeze(0).expand_as(img_tensor)