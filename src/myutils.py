#  myutils.py
# ============================================================================
# SIMPLE MULTI-MODEL OBJECT DETECTION SYSTEM
# Supports: YOLOv5s, YOLOv8n only
# ============================================================================

import os
import sys
from collections import defaultdict
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import piq
import requests
import torch
from PIL import Image
from piq import multi_scale_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = piq.LPIPS().to(device).eval()
dists_model = piq.DISTS().to(device).eval()

# Global variables - sẽ được set từ main.py
coco = None
data_dir = None
cat_id_to_index = None


def set_global_vars(coco_obj, data_dir_path, cat_mapping):
    """Set global variables từ main.py"""
    global coco, data_dir, cat_id_to_index
    coco = coco_obj
    data_dir = data_dir_path
    cat_id_to_index = cat_mapping


def compute_lpips(ori_tensor, adv_tensor):
    with torch.no_grad():
        return lpips_model(ori_tensor, adv_tensor).mean().item()


def compute_dists(ori_tensor, adv_tensor):
    with torch.no_grad():
        return dists_model(ori_tensor, adv_tensor).mean().item()


def compute_fsim(ori_tensor, adv_tensor):
    """Tính FSIM giữa hai tensor, sử dụng piq"""
    with torch.no_grad():
        return piq.fsim(ori_tensor, adv_tensor, data_range=1.0).item()


def compute_ms_ssim(ori_tensor, adv_tensor):
    # Ensure input is in [0, 1], shape (N, C, H, W)
    ms_ssim_val = multi_scale_ssim(ori_tensor, adv_tensor, data_range=1.0)
    return ms_ssim_val.item()


def deleteResultFolder(save_dir):
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            file_path = os.path.join(save_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)


def is_prediction_correct(pred_box, pred_cls, gt_boxes, gt_classes, iou_thresh=0.5):
    for gt_box, gt_cls in zip(gt_boxes, gt_classes):
        if pred_cls == gt_cls:
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_thresh:
                return True
    return False


def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_map_per_image(pred_boxes, pred_scores, pred_classes,
                          gt_boxes, gt_classes, iou_thresh=0.5):
    """
    Tính mAP cho 1 ảnh: trung bình AP theo từng lớp có mặt trong GT hoặc Pred.
    - pred_boxes: List of [x1, y1, x2, y2]
    - pred_scores: List of confidence score
    - pred_classes: List of int (class id)
    - gt_boxes: List of [x1, y1, x2, y2]
    - gt_classes: List of int (class id)
    """
    classwise_gt = defaultdict(list)
    classwise_pred = defaultdict(list)

    for box, cls in zip(gt_boxes, gt_classes):
        classwise_gt[cls].append(box)

    for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
        classwise_pred[cls].append((box, score))

    aps = []
    all_classes = set(classwise_gt.keys()).union(classwise_pred.keys())
    for cls in all_classes:
        gt_cls = classwise_gt[cls]
        preds_cls = classwise_pred[cls]
        if preds_cls:
            boxes, scores = zip(*preds_cls)
        else:
            boxes, scores = [], []
        ap = compute_ap(boxes, scores, gt_cls, iou_thresh=iou_thresh)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) == 0 else 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # ✅ Sort theo confidence giảm dần
    sorted_indices = np.argsort(pred_scores)[::-1]
    sorted_boxes = [pred_boxes[i] for i in sorted_indices]
    sorted_scores = [pred_scores[i] for i in sorted_indices]

    tp = []
    matched_gt = set()

    for pred_box in sorted_boxes:
        match_found = False
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if compute_iou(pred_box, gt_box) >= iou_thresh:
                match_found = True
                matched_gt.add(i)
                break
        tp.append(1 if match_found else 0)

    # Tính precision và recall
    tp_cumsum = np.cumsum(tp)
    recall = tp_cumsum / len(gt_boxes)
    precision = tp_cumsum / np.arange(1, len(tp) + 1)

    # Tính AP theo chuẩn COCO (11-point interpolation)
    return compute_ap_11_point(recall, precision)


def compute_ap_11_point(recall, precision):
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap


def compute_map_both(ori_tensor, gt_classes, adv_tensor, model, gt_boxes):
    """
    Compute mAP cho cả ảnh gốc và ảnh adversarial sử dụng BaseDetectionModel interface

    Args:
        ori_tensor: Original image tensor
        gt_classes: Ground truth classes
        adv_tensor: Adversarial image tensor
        model: BaseDetectionModel instance
        gt_boxes: Ground truth boxes

    Returns:
        map_orig, map_adv, pred_boxes_orig, pred_scores_orig, pred_labels_orig,
        pred_boxes_adv, pred_scores_adv, pred_labels_adv
    """

    # === Dự đoán ảnh gốc sử dụng model.predict() ===
    pred_boxes_orig, pred_scores_orig, pred_labels_orig = model.predict(ori_tensor)

    # Compute mAP cho ảnh gốc
    map_orig = compute_map_per_image(pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                                     gt_boxes, gt_classes, iou_thresh=0.5)

    # === Dự đoán ảnh adversarial sử dụng model.predict() ===
    pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_tensor)

    # Compute mAP cho ảnh adversarial
    map_adv = compute_map_per_image(pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                                    gt_boxes, gt_classes, iou_thresh=0.5)

    return map_orig, map_adv, pred_boxes_orig, pred_scores_orig, pred_labels_orig, pred_boxes_adv, pred_scores_adv, pred_labels_adv


def evaluate_model(ori_tensor, gt_classes, adv_tensor, model, gt_boxes):
    """
    Wrapper function để maintain backward compatibility
    """
    return compute_map_both(ori_tensor, gt_classes, adv_tensor, model, gt_boxes)

def exportImage(ori_tensor, gt_classes, adv_tensor, model, gt_boxes, img_id, save_dir,
                map_orig, map_adv, pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                variant_name="", csv_writer=None, exportOriFolder=False,
                exportAdvFolder=False):
    # === Tạo ảnh từ tensor ===
    # Lưu adv ảnh đúng format YOLO input (640x640)
    adv_np = adv_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    adv_img = Image.fromarray((adv_np * 255).astype(np.uint8))

    # === Lưu vào thư mục adv/ ===
    if exportAdvFolder:
        adv_dir = os.path.join(save_dir, "adv")
        os.makedirs(adv_dir, exist_ok=True)
        adv_dir = os.path.join(save_dir, "adv")
        os.makedirs(adv_dir, exist_ok=True)
        adv_path = os.path.join(adv_dir, f"{img_id}_{variant_name}.png")
        adv_img.save(adv_path)

    # === Nếu ảnh bị tấn công thành công, lưu thêm ảnh gốc vào ori/
    if exportOriFolder:
        ori_dir = os.path.join(save_dir, "ori")
        os.makedirs(ori_dir, exist_ok=True)
        ori_path = os.path.join(ori_dir, f"{img_id}_{variant_name}.png")
        ori_np = ori_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        ori_img = Image.fromarray((ori_np * 255).astype(np.uint8))
        ori_img.save(ori_path)

    # === Vẽ hình ===
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    font_size = 10

    # === ẢNH GỐC ===
    axs[0].imshow(ori_tensor.squeeze().detach().permute(1, 2, 0).cpu().numpy())
    axs[0].set_title(f"Original image\n(mAP {map_orig:.2f})", fontsize=30, y=1.02)
    axs[0].axis("off")

    for box, score, cls in zip(pred_boxes_orig, pred_scores_orig, pred_labels_orig):
        x1, y1, x2, y2 = box
        label = model.names[cls] if hasattr(model, 'names') else str(cls)
        color = 'green' if is_prediction_correct(box, cls, gt_boxes, gt_classes) else 'red'
        axs[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor=color, linestyle='--', fill=False, linewidth=2))
        # text

        axs[0].text(x1, y1 - 5, f"{label}\n{score:.2f}", fontsize=font_size,
                    color=color, backgroundcolor='white', verticalalignment='top')

    # === ẢNH ADV ===
    axs[1].imshow(adv_tensor.squeeze().detach().permute(1, 2, 0).cpu().numpy())
    axs[1].set_title(f"Adversarial example\n (mAP {map_adv:.2f})", fontsize=30, y=1.02)
    axs[1].axis("off")

    for box, score, cls in zip(pred_boxes_adv, pred_scores_adv, pred_labels_adv):
        x1, y1, x2, y2 = box
        label = model.names[cls] if hasattr(model, 'names') else str(cls)
        color = 'green' if is_prediction_correct(box, cls, gt_boxes, gt_classes) else 'red'
        axs[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor=color, linestyle='--', fill=False, linewidth=2))

        axs[1].text(x1, y1 - 5, f"{label}\n{score:.2f}", fontsize=font_size,
                    color=color, backgroundcolor='white', verticalalignment='top')

    # plt.suptitle(f'mAP before: {map_orig:.2f} | mAP after: {map_adv:.2f}', fontsize=30, y=1.02)
    compare_path = os.path.join(save_dir, f"compare_{img_id}_{variant_name}.png")
    plt.tight_layout()
    plt.savefig(compare_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"🖼️  So sánh ảnh {img_id} đã lưu tại: {compare_path}")
    if csv_writer is not None:
        # === Tính SSIM và PSNR giữa ảnh ori và adv ===
        ori_np = ori_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        adv_np = adv_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

        ssim_value = ssim_metric(ori_np, adv_np, data_range=1.0, channel_axis=-1)
        psnr_value = psnr_metric(ori_np, adv_np, data_range=1.0)
        ms_ssim_val = compute_ms_ssim(ori_tensor, adv_tensor)
        fsim_val = compute_fsim(ori_tensor, adv_tensor)
        lpips_val = compute_lpips(ori_tensor, adv_tensor)
        dists_val = compute_dists(ori_tensor, adv_tensor)

        # === Tính thêm L0 và L2 ===
        # Scale về 0–255 và convert sang int
        ori_uint8 = (ori_np * 255).astype(np.uint8)
        adv_uint8 = (adv_np * 255).astype(np.uint8)
        # L0: pixel khác ít nhất một channel
        L0 = np.sum(np.any(ori_uint8 != adv_uint8, axis=2))
        L2 = np.mean(np.linalg.norm(ori_np - adv_np, axis=2))

        # === Delta per-channel (mean |delta|) ===
        delta = np.abs(adv_np - ori_np)  # shape (H, W, 3)

        delta_R = float(np.mean(delta[:, :, 0]))
        delta_G = float(np.mean(delta[:, :, 1]))
        delta_B = float(np.mean(delta[:, :, 2]))


        row = [
            img_id, variant_name,
            map_orig, map_adv,
            ssim_value, psnr_value,
            ms_ssim_val, fsim_val,
            lpips_val, dists_val,
            L0, L2,
            delta_R, delta_G, delta_B
        ]

        csv_writer.writerow(row)

        # === In log chi tiết ===
        print(f"📄 {variant_name} - {img_id}:")
        print(f"  mAP: {round(map_orig, 4)} → {round(map_adv, 4)}")
        print(f"  SSIM={round(ssim_value, 4)}, PSNR={round(psnr_value, 4)}, MS-SSIM={round(ms_ssim_val, 4)}")
        print(f"  FSIM={round(fsim_val, 4)}, LPIPS={round(lpips_val, 4)}, DISTS={round(dists_val, 4)}")
        print(f"  L0 = {L0} pixels, L2 = {round(L2, 4)}")
        # print("----------------------------------------------------------")

    return map_orig, map_adv


def tensor_to_pil(img_tensor):
    img_np = img_tensor.squeeze().detach().cpu().numpy()
    img_np = (img_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(img_np)


def scale_box_with_letterbox(box, ratio, pad):
    x, y, w, h = box
    x1 = x * ratio[0] + pad[0]
    y1 = y * ratio[1] + pad[1]
    x2 = (x + w) * ratio[0] + pad[0]
    y2 = (y + h) * ratio[1] + pad[1]
    return [x1, y1, x2, y2]


def load_image_and_targets(img_id, model):
    global coco, data_dir, cat_id_to_index

    if coco is None or data_dir is None or cat_id_to_index is None:
        raise ValueError("Global variables chưa được khởi tạo. Gọi set_global_vars() trước.")

    # Load image từ COCO
    img_info = coco.loadImgs(img_id)[0]
    img_url = data_dir + img_info['file_name']
    response = requests.get(img_url)
    img_pil = Image.open(BytesIO(response.content)).convert('RGB')

    # Load annotations
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    if not anns:
        return None

    # === Sử dụng model interface để preprocess ===
    img_tensor = model.preprocess_input(img_pil)
    ori_tensor = img_tensor.clone().detach()

    # === Sử dụng model interface để prepare targets ===
    targets, gt_boxes, gt_classes = model.prepare_targets(anns, img_pil, cat_id_to_index)

    return img_tensor, ori_tensor, gt_boxes, gt_classes, targets


def evaluate_model(ori_tensor, gt_classes, adv_tensor, model, gt_boxes):
    """
    Wrapper function để maintain backward compatibility
    """
    return compute_map_both(ori_tensor, gt_classes, adv_tensor, model, gt_boxes)

def get_model_name(model):
    """
    Lấy tên model từ model object
    """
    if hasattr(model, 'model'):
        model_class = model.model.__class__.__name__
    else:
        model_class = model.__class__.__name__

    # Xử lý các tên model phổ biến
    if 'yolo' in model_class.lower():
        return 'yolov5'
    elif 'faster' in model_class.lower() and 'rcnn' in model_class.lower():
        return 'faster_rcnn'
    else:
        return model_class.lower()