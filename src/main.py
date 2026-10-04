# main.py
import os
import sys
# ✅ BỔ SUNG ĐƯỜNG DẪN VÀO sys.path
# Lấy đường dẫn thư mục gốc của project
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Lên 1 cấp từ src/ về freq/

# Thêm thư mục gốc vào sys.path để Python có thể tìm thấy module 'src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Thêm thư mục src vào sys.path (nếu cần)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

print(f"🔧 Project root: {project_root}")
print(f"🔧 Source dir: {src_dir}")
print(f"🔧 Python path: {sys.path[:3]}")  # In 3 đường dẫn đầu
import torch
from pycocotools.coco import COCO
import csv
import pandas as pd
from src.attack.fgsm import fgsm
from src.attack.pgd import pgd

from src.attack.gloattack import gloattack
from src.attack.gradient_based_dct4od import dct_mask
from src.attack.numod import numod
from src.attack.myconfig import MODEL_TYPE, NUM_IMAGE, SAVE_DIR, \
    GLOATTACK_CONFIGS, VISUALIZE_FREQUENCY, EXPORT_ADV_FOLDER, EXPORT_ORI_FOLDER, \
    RUN_ALL_MODELS, RUNTIME_MODELS, MODEL_DISPLAY_NAMES, IMAGE_LIST_DIR, \
    IMAGE_LIST_ALIASES, RUNTIME_CSV, RUNTIME_CSV_COLUMNS, \
    RUN_OURS, RUN_DCT_HIGH, RUN_DCT_MID, RUN_DCT_LOW, RUN_FGSM, RUN_NUMOD, RUN_PGD, \
    DCT_EPS, DCT_ITERS, FGSM_EPSILON, PGD_EPSILON, PGD_ITERS, PGD_ALPHA, NUMOD_CONFIGS
from src.myutils import (
    deleteResultFolder,
    load_image_and_targets,
    compute_map_per_image,
    set_global_vars, compute_iou
)

from src.models.model_registry import build_model

COCO_ANNOTATION_FILE = 'annotations/annotations/instances_val2017.json'
DATA_DIR = 'http://images.cocodataset.org/val2017/'

def append_image_id_to_cache(model_type, img_id):
    cache_file = get_cache_filename(model_type)

    # Đọc các ID đã có
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            existing_ids = {
                int(line.strip())
                for line in f
                if line.strip() and not line.startswith("#")
            }
    else:
        existing_ids = set()

    # Chỉ append nếu chưa tồn tại
    if img_id not in existing_ids:
        with open(cache_file, "a") as f:
            f.write(f"{img_id}\n")



def is_done(current_save_dir):
    """
    Check whether this configuration is already finished
    """
    return os.path.exists(os.path.join(current_save_dir, "done.txt"))


def mark_done(current_save_dir):
    """
    Mark this configuration as finished
    """
    done_path = os.path.join(current_save_dir, "done.txt")
    with open(done_path, "w") as f:
        f.write("DONE\n")

def print_defense_status(defense_enabled, defense_cfg):
    print("\n" + "=" * 60)
    if not defense_enabled:
        print("🛡️  DEFENSE STATUS: OFF")
        print("    • No adversarial defense is applied")
    else:
        print("🛡️  DEFENSE STATUS: ON")
        print("    • Inference-time defenses enabled:")
        for k, v in defense_cfg.items():
            if k == "jpeg":
                print(f"      - JPEG Compression (quality={v['quality']})")
            elif k == "feature_squeeze":
                print(f"      - Feature Squeezing (bit_depth={v['bit_depth']})")
            elif k == "gaussian":
                print(f"      - Gaussian Noise (sigma={v['sigma']})")
            elif k == "spatial_smooth":
                print(f"      - Spatial Smoothing (window_size={v['window_size']})")
            elif k == "randomized":
                print(f"      - Randomized Input Transform (p={v['p']})")
    print("=" * 60 + "\n")

# ===== THÊM VÀO ĐẦU main.py (sau các import) =====
def get_cache_filename(model_type):
    """
    Tạo tên file cache cho model
    Format: cache/{model}.txt
    """
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{model_type}.txt"
    return os.path.join(cache_dir, filename)


def load_cached_image_ids(model_type):
    """
    Load danh sách image IDs từ cache file nếu tồn tại

    Returns:
        list hoặc None nếu không tìm thấy cache
    """
    cache_file = get_cache_filename(model_type)

    if os.path.exists(cache_file):
        print(f"📂 Found cache file: {cache_file}")
        try:
            with open(cache_file, 'r') as f:
                # Bỏ qua các dòng comment (bắt đầu bằng #)
                img_ids = [int(line.strip()) for line in f
                           if line.strip() and not line.strip().startswith('#')]

            print(f"✅ Loaded {len(img_ids)} image IDs from cache")
            return img_ids

        except Exception as e:
            print(f"⚠️ Error reading cache file: {e}")
            return None
    else:
        print(f"Cache file not found: {cache_file}")
        return None


def save_image_ids_to_cache(model_type, img_ids):
    """
    Lưu danh sách image IDs vào cache file
    """
    cache_file = get_cache_filename(model_type)

    try:
        with open(cache_file, 'w') as f:
            f.write(f"# Model: {model_type}\n")
            f.write(f"# Num images: {len(img_ids)}\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\n")
            f.write("#" + "=" * 50 + "\n")
            for img_id in img_ids:
                f.write(f"{img_id}\n")
        print(f"💾 Saved {len(img_ids)} image IDs to cache: {cache_file}")
        return True
    except Exception as e:
        print(f"⚠️ Error saving cache file: {e}")
        return False

def filter_images_with_nonzero_map(model,img_ids,target_count,model_type,min_map_threshold=0.0001):
    """
    Filter images to only include those with non-zero mAP
    GUARANTEED to return exactly target_count images (or all available if not enough)

    Args:
        model: Detection model
        img_ids: List of image IDs to filter
        target_count: Number of images needed
        min_map_threshold: Minimum mAP threshold (default: 0.0001)

    Returns:
        List of filtered image IDs with non-zero mAP
    """
    print(f"🔍 Filtering images with mAP > {min_map_threshold}...")
    print(f"🎯 Target: {target_count} images")

    filtered_ids = []
    checked_count = 0

    for i, img_id in enumerate(img_ids):
        if len(filtered_ids) >= target_count:
            break

        checked_count += 1

        try:
            # Load and evaluate image
            result = load_image_and_targets(img_id, model)
            if result is None:
                continue

            img_tensor, ori_tensor, gt_boxes, gt_classes, targets = result

            # Get predictions and compute mAP
            pred_boxes_orig, pred_scores_orig, pred_labels_orig = model.predict(ori_tensor)
            map_orig = compute_map_per_image(
                pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                gt_boxes, gt_classes
            )

            if map_orig > min_map_threshold:
                filtered_ids.append(img_id)
                # 🔥 GHI CACHE NGAY LẬP TỨC
                append_image_id_to_cache(model_type, img_id)
                print(f"   ✅ Image {img_id}: mAP = {map_orig:.4f} (Found {len(filtered_ids)}/{target_count})")
            else:
                print(f"   ❌ Image {img_id}: mAP = {map_orig:.4f} (too low)")

            # Trong filter_images_with_nonzero_map, thêm debug:
            # Thêm vào compute_map_per_image hoặc filter_images_with_nonzero_map:
            if map_orig == 0.0:
                print(f"🔍 IoU Debug for image {img_id}:")

                # Tính IoU giữa predicted boxes và ground truth boxes
                for i, pred_box in enumerate(pred_boxes_orig[:3]):  # Check first 3 predictions
                    for j, gt_box in enumerate(gt_boxes[:3]):  # Check first 3 GT
                        iou = compute_iou(pred_box, gt_box)
                        if iou > 0.1:  # Only show meaningful IoUs
                            print(f"   Pred box {i} vs GT box {j}: IoU = {iou:.3f}")
                            print(
                                f"     Pred: {[round(x, 1) for x in pred_box]} (class {pred_labels_orig[i] if i < len(pred_labels_orig) else 'N/A'})")
                            print(
                                f"     GT:   {[round(x, 1) for x in gt_box]} (class {gt_classes[j] if j < len(gt_classes) else 'N/A'})")

        except Exception as e:
            print(f"   ⚠️ Error evaluating image {img_id}: {e}")
            continue

    print(f"📊 Summary: Checked {checked_count} images, found {len(filtered_ids)} with mAP > {min_map_threshold}")

    if len(filtered_ids) < target_count:
        print(f"⚠️ WARNING: Only found {len(filtered_ids)}/{target_count} images meeting criteria!")
        print(f"💡 Suggestion: Lower min_map_threshold or increase available images")
    else:
        print(f"✅ SUCCESS: Found exactly {target_count} images meeting criteria!")

    return filtered_ids[:target_count]  # Return exactly target_count (or all if less)

def get_csv_path(current_save_dir):
    """
    CSV name = _<folder_name>.csv
    """
    folder_name = os.path.basename(os.path.normpath(current_save_dir))
    return os.path.join(current_save_dir, f"_{folder_name}.csv")


def load_rq1_image_ids(model_type, n):
    """Top-n image IDs from result/rq1/1k images/<model>.txt."""
    filename = IMAGE_LIST_ALIASES.get(model_type, f"{model_type}.txt")
    path = os.path.join(project_root, IMAGE_LIST_DIR, filename)
    if not os.path.exists(path):
        fallback = os.path.join(project_root, IMAGE_LIST_DIR, f"{model_type}.txt")
        if fallback != path and os.path.exists(fallback):
            path = fallback
        else:
            raise FileNotFoundError(f"Image list not found for {model_type}: {path}")

    img_ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            img_ids.append(int(line))
            if len(img_ids) >= n:
                break
    print(f"📂 Loaded {len(img_ids)} image IDs from {path}")
    return img_ids


def mean_success_time(times):
    if not times:
        return ""
    return round(float(sum(times) / len(times)), 4)


def write_runtime_csv(runtime_rows):
    out_path = RUNTIME_CSV if os.path.isabs(RUNTIME_CSV) else os.path.join(project_root, RUNTIME_CSV)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(RUNTIME_CSV_COLUMNS)
        for row in runtime_rows:
            writer.writerow([row.get(col, "") for col in RUNTIME_CSV_COLUMNS])
    print(f"💾 Runtime table written: {out_path}")


def append_csv_averages(csv_path):
    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        return
    df = pd.read_csv(csv_path)
    if len(df) == 0 or len(df.columns) <= 2:
        return
    numeric_cols = df.columns[2:]
    averages = df[numeric_cols].mean().round(4)
    with open(csv_path, mode="a", newline="") as csv_file_append:
        csv_writer_append = csv.writer(csv_file_append)
        avg_row = ["AVERAGE", "summary"] + averages.tolist()
        csv_writer_append.writerow(avg_row)


def open_attack_csv(current_save_dir, extra_cols=None):
    os.makedirs(current_save_dir, exist_ok=True)
    deleteResultFolder(current_save_dir)
    csv_path = get_csv_path(current_save_dir)
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    header = [
        "image_id", "variant", "map_original", "map_adversarial",
        "SSIM", "PSNR", "MS-SSIM", "FSIM", "LPIPS", "DISTS", "L0", "L2",
    ]
    if extra_cols:
        header.extend(extra_cols)
    csv_writer.writerow(header)
    return csv_path, csv_file, csv_writer


if __name__ == '__main__':
    model_types = RUNTIME_MODELS if RUN_ALL_MODELS else [MODEL_TYPE]
    print("=" * 55)
    print(f"🤖 Models: {model_types}")
    print(f"📊 Number of images (top of 1k lists): {NUM_IMAGE}")
    print("=" * 55)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("📁 Loading COCO dataset...")
    coco = COCO(COCO_ANNOTATION_FILE)
    all_img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    cat_id_to_index = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    print(f"✅ COCO loaded: {len(all_img_ids)} images, {len(cat_ids)} categories")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    set_global_vars(coco, DATA_DIR, cat_id_to_index)

    runtime_rows = []
    for mt in model_types:
        display = MODEL_DISPLAY_NAMES.get(mt, mt)
        row = {col: "" for col in RUNTIME_CSV_COLUMNS}
        row["Model"] = display
        runtime_rows.append(row)

    write_runtime_csv(runtime_rows)

    for model_idx, model_type in enumerate(model_types):
        display_name = MODEL_DISPLAY_NAMES.get(model_type, model_type)
        print("\n" + "=" * 55)
        print(f"🤖 Model [{model_idx + 1}/{len(model_types)}]: {display_name} ({model_type})")
        print("=" * 55)

        sample_img_ids = load_rq1_image_ids(model_type, NUM_IMAGE)
        print(f"✅ Ready to attack with {len(sample_img_ids)} images")
        print(f"📋 IDs: {sample_img_ids}")

        model = build_model(model_type, device)
        row = runtime_rows[model_idx]

        if RUN_OURS:
            print(f"\n🎯 Starting attacks on {model.__class__.__name__}...")
            for config in GLOATTACK_CONFIGS:
                epsilon = config['epsilon']
                max_iters = config['max_iters']
                interval = config['interval']
                print(f"\n⚡ Running with epsilon={epsilon}, max_iters={max_iters}")
                current_save_dir = f"{SAVE_DIR}/{model_type}_gloattack_eps{epsilon:.4f}_iter{max_iters}"
                csv_path, csv_file, csv_writer = open_attack_csv(
                    current_save_dir, extra_cols=["delta_R", "delta_G", "delta_B"]
                )
                success, times = gloattack(
                    model, sample_img_ids, epsilon=epsilon, max_iters=max_iters,
                    interval=interval, csv_writer=csv_writer, save_dir=current_save_dir,
                    visualize_frequency=VISUALIZE_FREQUENCY, exportAdvFolder=EXPORT_ADV_FOLDER,
                    exportOriFolder=EXPORT_ORI_FOLDER)
                print(f"✅ eps={epsilon}, iter={max_iters}: {success}/{len(sample_img_ids)} successful attacks")
                csv_file.close()
                append_csv_averages(csv_path)
                mark_done(current_save_dir)
                row["Ours"] = mean_success_time(times)
            print("\n🎉 attacks completed!")

        if RUN_DCT_HIGH or RUN_DCT_MID or RUN_DCT_LOW:
            print(f"\n🎯 Starting DCT Mask attacks on {model.__class__.__name__}...")
            dct_configs = []
            if RUN_DCT_LOW:
                dct_configs.append({'mask_type': 'low', 'eps': DCT_EPS, 'iters': DCT_ITERS, 'col': 'DCT_low'})
            if RUN_DCT_HIGH:
                dct_configs.append({'mask_type': 'high', 'eps': DCT_EPS, 'iters': DCT_ITERS, 'col': 'DCT_high'})
            if RUN_DCT_MID:
                dct_configs.append({'mask_type': 'mid', 'eps': DCT_EPS, 'iters': DCT_ITERS, 'col': 'DCT_mid'})

            for config in dct_configs:
                mask_type = config['mask_type']
                eps = config['eps']
                iters = config['iters']
                print(f"\n⚡ Running DCT Mask with {mask_type} frequency, eps={eps}")
                current_save_dir = f"{SAVE_DIR}/{model_type}_dct_{mask_type}_eps{ep:.4f}_iter{iters}"
                csv_path, csv_file, csv_writer = open_attack_csv(current_save_dir)
                success, times = dct_mask(
                    model, sample_img_ids, mask_type=mask_type, eps=eps, iters=iters,
                    csv_writer=csv_writer, save_dir=current_save_dir)
                print(f"✅ DCT Mask {mask_type}: {success}/{len(sample_img_ids)} successful attacks")
                csv_file.close()
                append_csv_averages(csv_path)
                mark_done(current_save_dir)
                row[config['col']] = mean_success_time(times)
            print("\n🎉 DCT Mask attacks completed!")

        if RUN_FGSM:
            print(f"\n🎯 Starting FGSM attacks on {model.__class__.__name__}...")
            ep = FGSM_EPSILON
            print(f"\n⚡ Running FGSM with epsilon={ep}")
            current_save_dir = f"{SAVE_DIR}/{model_type}_fgsm_ep{ep:.4f}"
            csv_path, csv_file, csv_writer = open_attack_csv(current_save_dir)
            success, times = fgsm(
                model, sample_img_ids, epsilon=ep, csv_writer=csv_writer, save_dir=current_save_dir)
            print(f"✅ FGSM epsilon={ep}: {success}/{len(sample_img_ids)} successful attacks")
            csv_file.close()
            append_csv_averages(csv_path)
            mark_done(current_save_dir)
            row["FGSM"] = mean_success_time(times)
            print("\n🎉 All attacks completed!")

        if RUN_NUMOD:
            print(f"\n🎯 Starting NumbOD attacks on {model.__class__.__name__}...")
            for config in NUMOD_CONFIGS:
                epsilon = config['epsilon']
                max_iters = config['max_iters']
                alpha = config['alpha']
                lambda_sf = config['lambda_sf']
                print(f"\n⚡ Running NumbOD with epsilon={epsilon}, max_iters={max_iters}")
                current_save_dir = f"{SAVE_DIR}/{model_type}_numod_eps{epsilon:.4f}_iter{max_iters}"
                csv_path, csv_file, csv_writer = open_attack_csv(current_save_dir)
                success, times = numod(
                    model, sample_img_ids, epsilon=epsilon, max_iters=max_iters,
                    alpha=alpha, lambda_sf=lambda_sf,
                    csv_writer=csv_writer, save_dir=current_save_dir)
                print(f"✅ NumbOD eps={epsilon}, iter={max_iters}: {success}/{len(sample_img_ids)} successful attacks")
                csv_file.close()
                append_csv_averages(csv_path)
                mark_done(current_save_dir)
                row["NumbOD"] = mean_success_time(times)
            print("\n🎉 NumbOD attacks completed!")

        if RUN_PGD:
            print(f"\n🎯 Starting PGD attacks on {model.__class__.__name__}...")
            ep = PGD_EPSILON
            print(f"\n⚡ Running PGD with epsilon={ep}, iter={PGD_ITERS}")
            current_save_dir = f"{SAVE_DIR}/{model_type}_pgd_ep{ep:.4f}_iter{PGD_ITERS}"
            csv_path, csv_file, csv_writer = open_attack_csv(current_save_dir)
            success, times = pgd(
                model, sample_img_ids, epsilon=ep, alpha=PGD_ALPHA, num_iter=PGD_ITERS,
                random_start=True, csv_writer=csv_writer, save_dir=current_save_dir)
            print(f"✅ PGD epsilon={ep}: {success}/{len(sample_img_ids)} successful attacks")
            csv_file.close()
            append_csv_averages(csv_path)
            mark_done(current_save_dir)
            row["PGD"] = mean_success_time(times)
            print("\n🎉 All attacks completed!")

        write_runtime_csv(runtime_rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n📁 Runtime summary: {RUNTIME_CSV}")
