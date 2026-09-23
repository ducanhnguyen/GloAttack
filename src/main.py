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
    GLOATTACK_CONFIGS, VISUALIZE_FREQUENCY, EXPORT_ADV_FOLDER, EXPORT_ORI_FOLDER
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


if __name__ == '__main__':
    print("=" * 55)
    print(f"🤖 Selected Model: {MODEL_TYPE}")
    print(f"📊 Number of images: {NUM_IMAGE}")
    print("=" * 55)

    # Tạo thư mục lưu ảnh nếu chưa tồn tại
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Khởi tạo COCO API
    print("📁 Loading COCO dataset...")
    coco = COCO(COCO_ANNOTATION_FILE)
    all_img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()  # Map category_id gốc COCO về index 0–79
    cat_id_to_index = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    print(f"✅ COCO loaded: {len(all_img_ids)} images, {len(cat_ids)} categories")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    model = build_model(MODEL_TYPE, device)

    set_global_vars(coco, DATA_DIR, cat_id_to_index)

    # Get valid images with ground-truth
    valid_img_ids = [img_id for img_id in all_img_ids if coco.getAnnIds(imgIds=img_id)]
    print(f"🔍 Total images with GT: {len(valid_img_ids)}/{len(all_img_ids)}")

    # 🆕 ===== CACHE MECHANISM (CHỈ PHỤ THUỘC MODEL) =====
    print("\n" + "=" * 55)
    print("🔍 CHECKING IMAGE ID CACHE")
    print("=" * 55)

    # Try load from cache first
    cached_img_ids = load_cached_image_ids(MODEL_TYPE)

    if cached_img_ids is None or len(cached_img_ids) == 0:
        cache_file = get_cache_filename(MODEL_TYPE)

        with open(cache_file, "w") as f:
            f.write(f"# Model: {MODEL_TYPE}\n")
            f.write(f"# Target images: {NUM_IMAGE}\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\n")
            f.write("#" + "=" * 50 + "\n")


        # Cache không tồn tại → Filter images
        print(f"\n🔄 Starting image filtering for {MODEL_TYPE}...")
        print(f"   Target: {NUM_IMAGE} images with mAP > 0.001")
        print(f"   This may take several minutes...\n")

        # Create model để filter
        model = build_model(MODEL_TYPE, device)

        # Filter images with non-zero mAP
        sample_img_ids = filter_images_with_nonzero_map(
            model, valid_img_ids, NUM_IMAGE,
            MODEL_TYPE,
            min_map_threshold=0.1
        )

        # Save to cache
        print(f"✅ Cache updated incrementally during filtering")

        # if len(sample_img_ids) > 0:
        #     save_image_ids_to_cache(MODEL_TYPE, sample_img_ids)
        #     print(f"✅ Cache saved for future runs!")
        # else:
        #     print(f"⚠️ No valid images found!")
        #     exit(1)
    else:
        # Cache tồn tại → Sử dụng cached IDs (lấy NUM_IMAGE đầu tiên)
        print(f"✅ Found {len(cached_img_ids)} cached image IDs")

        if len(cached_img_ids) >= NUM_IMAGE:
            sample_img_ids = cached_img_ids[:NUM_IMAGE]
            print(f"⏩ Using first {NUM_IMAGE} images from cache")
        else:
            print(f"⚠️ Cache has only {len(cached_img_ids)} images, need {NUM_IMAGE}")
            print(f"🔄 Will filter additional images...")

            # Tạo model và filter thêm
            model = build_model(MODEL_TYPE, device)

            # Loại bỏ các IDs đã có trong cache
            remaining_img_ids = [img_id for img_id in valid_img_ids
                                 if img_id not in cached_img_ids]

            # Filter thêm
            additional_needed = NUM_IMAGE - len(cached_img_ids)
            additional_img_ids = filter_images_with_nonzero_map(
                model, remaining_img_ids, additional_needed,
                MODEL_TYPE,
                min_map_threshold=0.001
            )

            # Gộp lại
            sample_img_ids = cached_img_ids + additional_img_ids

            # Cập nhật cache
            # if len(sample_img_ids) >= NUM_IMAGE:
            #     save_image_ids_to_cache(MODEL_TYPE, sample_img_ids)
            #     sample_img_ids = sample_img_ids[:NUM_IMAGE]
            sample_img_ids = sample_img_ids[:NUM_IMAGE]
            print("✅ Cache updated incrementally (resume-safe)")

        # Tạo model sau khi đã có IDs
        if 'model' not in locals():
            print(f"🔧 Creating model: {MODEL_TYPE}...")
            model = build_model(MODEL_TYPE, device)

    print("=" * 55)
    print(f"✅ Ready to attack with {len(sample_img_ids)} images")
    print(f"📋 First 10 IDs: {sample_img_ids[:10]}")
    print("=" * 55 + "\n")

    # Run our attacks
    if False:
        print(f"\n🎯 Starting attacks on {model.__class__.__name__}...")

        for config in GLOATTACK_CONFIGS:
            epsilon = config['epsilon']
            max_iters = config['max_iters']
            interval = config['interval']

            print(f"\n⚡ Running with epsilon={epsilon}, max_iters={max_iters}")

            # Create save directory

            current_save_dir = f"{SAVE_DIR}/{MODEL_TYPE}_gloattack_eps{epsilon}_iter{max_iters}"

            os.makedirs(current_save_dir, exist_ok=True)
            if is_done(current_save_dir):
                print(f"⏭️  Skip (done): {current_save_dir}")
                continue

            deleteResultFolder(current_save_dir)

            # Setup CSV logging
            # csv_file = open(os.path.join(current_save_dir, "map_results.csv"), mode='w', newline='')
            csv_path = get_csv_path(current_save_dir)
            csv_file = open(csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "image_id", "variant", "map_original", "map_adversarial",
                "SSIM", "PSNR", "MS-SSIM", "FSIM", "LPIPS", "DISTS", "L0", "L2",
                "delta_R", "delta_G", "delta_B"
            ])

            success = gloattack(model, sample_img_ids, epsilon=epsilon, max_iters=max_iters,
                                interval=interval, csv_writer=csv_writer, save_dir=current_save_dir,
                                visualize_frequency=VISUALIZE_FREQUENCY, exportAdvFolder=EXPORT_ADV_FOLDER,
                                exportOriFolder=EXPORT_ORI_FOLDER)
            print(f"✅ eps={epsilon}, iter={max_iters}: {success}/{len(sample_img_ids)} successful attacks")

            # === TÍNH VÀ GHI HÀNG AVERAGE ===
            csv_file.close()
            # csv_path = os.path.join(current_save_dir, "map_results.csv")
            # csv_path = get_csv_path(current_save_dir)

            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                df = pd.read_csv(csv_path)
                if len(df) > 0 and len(df.columns) > 2:
                    # Tính average cho các cột từ map_original trở đi (cột thứ 2 trở đi)
                    numeric_cols = df.columns[2:]  # Bỏ qua "image_id" và "variant"
                    averages = df[numeric_cols].mean().round(4)

                    # Mở lại file để ghi thêm hàng average
                    with open(csv_path, mode='a', newline='') as csv_file_append:
                        csv_writer_append = csv.writer(csv_file_append)
                        avg_row = ["AVERAGE", "summary"] + averages.tolist()
                        csv_writer_append.writerow(avg_row)
            csv_file.close()
            mark_done(current_save_dir)

        print("\n🎉 attacks completed!")
        print(f"📁 Results saved in: {MODEL_TYPE}_gloattack_* folders")

    # Run FGSM attacks
    if False:
        print(f"\n🎯 Starting FGSM attacks on {model.__class__.__name__}...")
        #EPSILON_VALUES = [0.004, 0.008, 0.012, 0.016, 0.020, 0.024]
        EPSILON_VALUES = [0.012]
        for ep in EPSILON_VALUES:
            print(f"\n⚡ Running FGSM with epsilon={ep}")

            # Create save directory for this epsilon
            current_save_dir = (
                f"{SAVE_DIR}/"
                f"{MODEL_TYPE}_fgsm_ep{ep}_iter{max_iters}"
            )
            os.makedirs(current_save_dir, exist_ok=True)
            if is_done(current_save_dir):
                print(f"⏭️  Skip (done): {current_save_dir}")
                continue

            deleteResultFolder(current_save_dir)

            # Setup CSV logging
            csv_path = get_csv_path(current_save_dir)
            csv_file = open(csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            # csv_file = open(os.path.join(current_save_dir, "map_results.csv"), mode='w', newline='')
            # csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "image_id", "variant", "map_original", "map_adversarial",
                "SSIM", "PSNR", "MS-SSIM", "FSIM", "LPIPS", "DISTS", "L0", "L2"
            ])

            success = fgsm(model, sample_img_ids, epsilon=ep, csv_writer=csv_writer, save_dir=current_save_dir)
            print(f"✅ FGSM epsilon={ep}: {success}/{len(sample_img_ids)} successful attacks")

            # === TÍNH VÀ GHI HÀNG AVERAGE ===
            csv_file.close()
            # csv_path = os.path.join(current_save_dir, "map_results.csv")
            # csv_path = get_csv_path(current_save_dir)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                df = pd.read_csv(csv_path)
                if len(df) > 0 and len(df.columns) > 2:
                    # Tính average cho các cột từ map_original trở đi (cột thứ 2 trở đi)
                    numeric_cols = df.columns[2:]  # Bỏ qua "image_id" và "variant"
                    averages = df[numeric_cols].mean().round(4)

                    # Mở lại file để ghi thêm hàng average
                    with open(csv_path, mode='a', newline='') as csv_file_append:
                        csv_writer_append = csv.writer(csv_file_append)
                        avg_row = ["AVERAGE", "summary"] + averages.tolist()
                        csv_writer_append.writerow(avg_row)
            csv_file.close()
            mark_done(current_save_dir)

        print("\n🎉 All attacks completed!")
        print(f"📁 Results saved in: {MODEL_TYPE}_fgsm_ep* folders")

    # PGD
    if False:
        print(f"\n🎯 Starting PGD attacks on {model.__class__.__name__}...")
        #EPSILON_VALUES = [0.004, 0.008, 0.012, 0.016, 0.020, 0.024]
        EPSILON_VALUES = [0.008]
        MAX_ITER = 100
        for ep in EPSILON_VALUES:
            print(f"\n⚡ Running PGD with epsilon={ep}")

            # Create save directory for this epsilon
            current_save_dir = (
                f"{SAVE_DIR}/"
                f"{MODEL_TYPE}_pgd_ep{ep}_iter{max_iters}"
            )
            os.makedirs(current_save_dir, exist_ok=True)
            if is_done(current_save_dir):
                print(f"⏭️  Skip (done): {current_save_dir}")
                continue

            deleteResultFolder(current_save_dir)

            # Setup CSV logging
            # csv_file = open(os.path.join(current_save_dir, "map_results.csv"), mode='w', newline='')
            # csv_writer = csv.writer(csv_file)
            csv_path = get_csv_path(current_save_dir)
            csv_file = open(csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "image_id", "variant", "map_original", "map_adversarial",
                "SSIM", "PSNR", "MS-SSIM", "FSIM", "LPIPS", "DISTS", "L0", "L2"
            ])

            success = pgd(model, sample_img_ids, epsilon=ep, alpha=0.005, num_iter=MAX_ITER, random_start=True,
                          csv_writer=csv_writer, save_dir=current_save_dir)
            print(f"✅ PGD epsilon={ep}: {success}/{len(sample_img_ids)} successful attacks")

            # === TÍNH VÀ GHI HÀNG AVERAGE ===
            csv_file.close()
            # csv_path = os.path.join(current_save_dir, "map_results.csv")
            # csv_path = get_csv_path(current_save_dir)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                df = pd.read_csv(csv_path)
                if len(df) > 0 and len(df.columns) > 2:
                    # Tính average cho các cột từ map_original trở đi (cột thứ 2 trở đi)
                    numeric_cols = df.columns[2:]  # Bỏ qua "image_id" và "variant"
                    averages = df[numeric_cols].mean().round(4)

                    # Mở lại file để ghi thêm hàng average
                    with open(csv_path, mode='a', newline='') as csv_file_append:
                        csv_writer_append = csv.writer(csv_file_append)
                        avg_row = ["AVERAGE", "summary"] + averages.tolist()
                        csv_writer_append.writerow(avg_row)
            csv_file.close()
            mark_done(current_save_dir)

        print("\n🎉 All attacks completed!")
        print(f"📁 Results saved in: {MODEL_TYPE}_pgd_ep* folders")

    # Run DCT Mask attacks
    if False:
        print(f"\n🎯 Starting DCT Mask attacks on {model.__class__.__name__}...")
        DCT_CONFIGS = [
            {'mask_type': 'low', 'eps': 1/255, 'iters': 1000},
            {'mask_type': 'high', 'eps': 1/255, 'iters': 1000},
            {'mask_type': 'mid', 'eps': 1/255, 'iters': 1000}
        ]

        for config in DCT_CONFIGS:
            mask_type = config['mask_type']
            eps = config['eps']
            iters = config['iters']

            print(f"\n⚡ Running DCT Mask with {mask_type} frequency, eps={eps}")

            current_save_dir = (
                f"{SAVE_DIR}/"
                f"{MODEL_TYPE}_dct_{mask_type}_eps{eps}_iter{iters}"
            )
            os.makedirs(current_save_dir, exist_ok=True)
            if is_done(current_save_dir):
                print(f"⏭️  Skip (done): {current_save_dir}")
                continue

            deleteResultFolder(current_save_dir)

            # csv_file = open(os.path.join(current_save_dir, "map_results.csv"), mode='w', newline='')
            csv_path = get_csv_path(current_save_dir)
            csv_file = open(csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "image_id", "variant", "map_original", "map_adversarial",
                "SSIM", "PSNR", "MS-SSIM", "FSIM", "LPIPS", "DISTS", "L0", "L2"
            ])

            success = dct_mask(model, sample_img_ids, mask_type=mask_type, eps=eps, iters=iters,
                               csv_writer=csv_writer, save_dir=current_save_dir)
            print(f"✅ DCT Mask {mask_type}: {success}/{len(sample_img_ids)} successful attacks")

            # === TÍNH VÀ GHI HÀNG AVERAGE ===
            csv_file.close()

            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                df = pd.read_csv(csv_path)
                if len(df) > 0 and len(df.columns) > 2:
                    # Tính average cho các cột từ map_original trở đi (cột thứ 2 trở đi)
                    numeric_cols = df.columns[2:]  # Bỏ qua "image_id" và "variant"
                    averages = df[numeric_cols].mean().round(4)

                    # Mở lại file để ghi thêm hàng average
                    with open(csv_path, mode='a', newline='') as csv_file_append:
                        csv_writer_append = csv.writer(csv_file_append)
                        avg_row = ["AVERAGE", "summary"] + averages.tolist()
                        csv_writer_append.writerow(avg_row)
            csv_file.close()
            mark_done(current_save_dir)

        print("\n🎉 DCT Mask attacks completed!")

    # Run NumbOD attacks
    if True:
        print(f"\n🎯 Starting NumbOD attacks on {model.__class__.__name__}...")
        NUMOD_CONFIGS = [
            #{'epsilon': 0.01, 'max_iters': 500, 'alpha': 0.5, 'lambda_sf': 0.5},
            #{'epsilon': 0.03, 'max_iters': 500, 'alpha': 0.5, 'lambda_sf': 0.5},
            #{'epsilon': 0.05, 'max_iters': 500, 'alpha': 0.5, 'lambda_sf': 0.5},
            #{'epsilon': 0.07, 'max_iters': 500, 'alpha': 0.5, 'lambda_sf': 0.5},
            {'epsilon': 0.09, 'max_iters': 500, 'alpha': 0.5, 'lambda_sf': 0.5}
        ]

        for config in NUMOD_CONFIGS:
            epsilon = config['epsilon']
            max_iters = config['max_iters']
            alpha = config['alpha']
            lambda_sf = config['lambda_sf']

            print(f"\n⚡ Running NumbOD with epsilon={epsilon}, max_iters={max_iters}")

            # Create save directory
            current_save_dir = (
                f"{SAVE_DIR}/"
                f"{MODEL_TYPE}_numod_eps{epsilon}_iter{max_iters}"
            )
            os.makedirs(current_save_dir, exist_ok=True)
            if is_done(current_save_dir):
                print(f"⏭️  Skip (done): {current_save_dir}")
                continue

            deleteResultFolder(current_save_dir)

            # Setup CSV logging
            # csv_file = open(os.path.join(current_save_dir, "map_results.csv"), mode='w', newline='')
            # csv_writer = csv.writer(csv_file)
            csv_path = get_csv_path(current_save_dir)
            csv_file = open(csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "image_id", "variant", "map_original", "map_adversarial",
                "SSIM", "PSNR", "MS-SSIM", "FSIM", "LPIPS", "DISTS", "L0", "L2"
            ])

            success = numod(model, sample_img_ids, epsilon=epsilon, max_iters=max_iters,
                            alpha=alpha, lambda_sf=lambda_sf,
                            csv_writer=csv_writer, save_dir=current_save_dir)
            print(f"✅ NumbOD eps={epsilon}, iter={max_iters}: {success}/{len(sample_img_ids)} successful attacks")

            # === TÍNH VÀ GHI HÀNG AVERAGE ===
            csv_file.close()
            # csv_path = os.path.join(current_save_dir, "map_results.csv")
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                df = pd.read_csv(csv_path)
                if len(df) > 0 and len(df.columns) > 2:
                    # Tính average cho các cột từ map_original trở đi (cột thứ 2 trở đi)
                    numeric_cols = df.columns[2:]  # Bỏ qua "image_id" và "variant"
                    averages = df[numeric_cols].mean().round(4)

                    # Mở lại file để ghi thêm hàng average
                    with open(csv_path, mode='a', newline='') as csv_file_append:
                        csv_writer_append = csv.writer(csv_file_append)
                        avg_row = ["AVERAGE", "summary"] + averages.tolist()
                        csv_writer_append.writerow(avg_row)
            csv_file.close()
            mark_done(current_save_dir)

        print("\n🎉 NumbOD attacks completed!")
        print(f"📁 Results saved in: {MODEL_TYPE}_numod_* folders")