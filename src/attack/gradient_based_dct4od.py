import torch
import numpy as np
from scipy.fftpack import dct, idct
from src.myutils import (
    load_image_and_targets, exportImage, compute_map_per_image
)


def dct_mask(model, sample_img_ids, mask_type='low', eps=16 / 255, iters=10, alpha=None,
             attack_mode='white_box', targeted=False, csv_writer=None, save_dir="out"):
    """
    DCT mask attack theo paper "On the Effectiveness of Low Frequency Perturbations"

    Args:
        model: BaseDetectionModel instance
        sample_img_ids: List of image IDs
        mask_type: Loại mask ('low', 'mid', 'high', 'random') - Figure 1
        eps: Maximum perturbation bound
        iters: Số iteration
        alpha: Learning rate (nếu None thì alpha = eps/iters)
        attack_mode: 'white_box' or 'black_box'
        targeted: True for targeted attack, False for non-targeted
        csv_writer: CSV writer for logging
        save_dir: Directory to save results
    """
    success = 0
    print(f"🎯 Starting DCT mask attack with mask_type={mask_type}, eps={eps}, iters={iters}")
    print(f"   Mode: {'Targeted' if targeted else 'Non-targeted'} {attack_mode}")

    if alpha is None:
        alpha = eps / iters

    total_images = len(sample_img_ids)
    for idx, img_id in enumerate(sample_img_ids, 1):
        print("=" * 50)
        print(f"Image {img_id} [{idx:2d}/{total_images}]")

        try:
            # === LOAD IMAGE VÀ TARGETS ===
            result = load_image_and_targets(img_id, model)
            if result is None:
                print(f"Skipping image {img_id}: No valid annotations")
                continue
            img_tensor, ori_tensor, gt_boxes, gt_classes, targets = result

            # === Baseline evaluation ===
            pred_boxes_orig, pred_scores_orig, pred_labels_orig = model.predict(ori_tensor)
            map_orig = compute_map_per_image(
                pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                gt_boxes, gt_classes
            )

            # === DCT Mask Attack theo paper ===
            # Khởi tạo từ ảnh gốc
            adv_img = img_tensor.clone().detach()

            # Tạo DCT frequency mask một lần (Figure 1)
            B, C, H, W = img_tensor.shape
            freq_mask = create_dct_frequency_mask(H, W, mask_type).to(img_tensor.device)
            freq_mask = freq_mask[None, None, :, :].repeat(B, C, 1, 1)  # (B, C, H, W)

            attack_successful = False
            final_iteration = 0

            for i in range(iters):
                adv_img.requires_grad_(True)

                # Forward pass để tính loss
                if targeted and len(pred_labels_orig) > 0:
                    # Targeted attack: try to make model predict wrong class
                    target_class = (pred_labels_orig[0] + 1) % len(model.names)
                    loss = -model.compute_loss(adv_img, targets)  # Minimize correct class
                else:
                    # Non-targeted attack: maximize loss
                    loss = model.compute_loss(adv_img, targets)

                if loss is None or not isinstance(loss, torch.Tensor):
                    print(f"Skipping iteration {i} for image {img_id}: Invalid loss")
                    break

                # Backward để tính gradient
                loss.backward()

                # === DCT-based frequency domain attack ===
                grad = adv_img.grad.detach()

                # Apply DCT frequency mask theo paper methodology
                masked_grad = apply_dct_frequency_mask(grad, freq_mask)

                # Cập nhật theo gradient ascent/descent
                with torch.no_grad():
                    if targeted:
                        adv_img = adv_img - alpha * masked_grad.sign()  # Gradient descent
                    else:
                        adv_img = adv_img + alpha * masked_grad.sign()  # Gradient ascent

                    # Project về epsilon ball (L∞ constraint) - Equation theo paper
                    perturbation = adv_img - img_tensor
                    perturbation = torch.clamp(perturbation, -eps, eps)
                    adv_img = img_tensor + perturbation

                    # Clamp về [0,1]
                    adv_img = torch.clamp(adv_img, 0, 1)

                # === CHECK SUCCESS AFTER EACH ITERATION ===
                pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_img)
                map_adv = compute_map_per_image(
                    pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                    gt_boxes, gt_classes
                )

                print(f"Iter {i + 1}/{iters}: mAP {map_orig:.4f} → {map_adv:.4f}")

                # Check if attack successful
                success_condition = map_adv < map_orig if not targeted else (
                        len(pred_labels_adv) > 0 and pred_labels_adv[0] != pred_labels_orig[0]
                )

                if success_condition:
                    attack_successful = True
                    final_iteration = i + 1
                    print(f"Attack successful at iteration {final_iteration}!")
                    break

            # === FINAL EVALUATION AND EXPORT ===
            if attack_successful:
                # Get final predictions for export
                pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_img)
                map_adv = compute_map_per_image(
                    pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                    gt_boxes, gt_classes
                )

                mode_str = f"{'targeted' if targeted else 'nontargeted'}_{attack_mode}"
                variant_name = f"dct_{mask_type}_{mode_str}_eps{eps}_iter{final_iteration}"
                exportImage(ori_tensor, gt_classes, adv_img, model,
                            gt_boxes, img_id, save_dir, map_orig, map_adv,
                            pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                            pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                            variant_name=variant_name, csv_writer=csv_writer)
                success += 1
                print(f"Attack successful on image {img_id} after {final_iteration} iterations")
            else:
                print(f"Attack failed on image {img_id} after {iters} iterations")

        except Exception as e:
            print(f"Error attacking image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # === THỐNG KÊ KẾT QUẢ ===
    success_rate = (success / len(sample_img_ids)) * 100 if len(sample_img_ids) > 0 else 0
    mode_str = f"{'Targeted' if targeted else 'Non-targeted'} {attack_mode}"
    print(f"DCT Mask {mask_type} ({mode_str}) {model.__class__.__name__}: "
          f"Successful attacks: {success}/{len(sample_img_ids)} ({success_rate:.2f}%)")
    return success


def create_dct_frequency_mask(H, W, mask_type='low', keep_ratio=0.2):
    mask = torch.zeros(H, W, dtype=torch.float32)

    if mask_type == 'low':
        # DCT_Low: Top-left corner (low frequencies)
        # Paper Figure 1 shows concentrated low-freq region
        low_h = int(H * np.sqrt(keep_ratio))
        low_w = int(W * np.sqrt(keep_ratio))
        mask[:low_h, :low_w] = 1.0

    elif mask_type == 'high':
        # DCT_High: High frequencies (complement of low)
        low_h = int(H * np.sqrt(keep_ratio))
        low_w = int(W * np.sqrt(keep_ratio))
        mask[low_h:, :] = 1.0
        mask[:, low_w:] = 1.0

    elif mask_type == 'mid':
        # DCT_Mid: Mid-frequency band
        # Create band-pass filter in DCT domain
        # Based on Figure 1 pattern
        center_h, center_w = H // 2, W // 2

        # Create concentric bands
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        # Distance from top-left (DC component in DCT)
        dist_from_dc = torch.sqrt(y.float() ** 2 + x.float() ** 2)
        max_dist = torch.sqrt(torch.tensor(H ** 2 + W ** 2, dtype=torch.float32))
        norm_dist = dist_from_dc / max_dist

        # Mid-frequency band
        low_thresh = 0.2
        high_thresh = 0.6
        mask = ((norm_dist >= low_thresh) & (norm_dist <= high_thresh)).float()

    elif mask_type == 'random':
        # DCT_Random: Random frequency components
        total_components = H * W
        num_components = int(keep_ratio * total_components)

        # Random selection
        flat_indices = torch.randperm(total_components)[:num_components]
        mask_flat = mask.view(-1)
        mask_flat[flat_indices] = 1.0

    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return mask


def apply_dct_frequency_mask(grad, freq_mask):
    B, C, H, W = grad.shape
    masked_grad = torch.zeros_like(grad)

    for b in range(B):
        for c in range(C):
            # Convert to numpy for DCT processing
            grad_np = grad[b, c].detach().cpu().numpy()
            mask_np = freq_mask[b, c].detach().cpu().numpy()

            # Apply 2D DCT
            grad_dct = dct(dct(grad_np.T, norm='ortho').T, norm='ortho')

            # Apply frequency mask
            masked_dct = grad_dct * mask_np

            # Inverse DCT
            masked_grad_np = idct(idct(masked_dct.T, norm='ortho').T, norm='ortho')

            # Convert back to tensor
            masked_grad[b, c] = torch.from_numpy(masked_grad_np).to(grad.device)

    return masked_grad


def visualize_dct_masks(H=64, W=64, save_path="dct_masks_visualization.png"):
    """
    Visualize DCT frequency masks
    """
    import matplotlib.pyplot as plt

    mask_types = ['low', 'mid', 'high', 'random']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, mask_type in enumerate(mask_types):
        mask = create_dct_frequency_mask(H, W, mask_type)

        im = axes[i].imshow(mask.numpy(), cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(f'DCT_{mask_type.capitalize()}', fontsize=14)
        axes[i].axis('off')

        # Add colorbar
        plt.colorbar(im, ax=axes[i], shrink=0.8)

    plt.suptitle('DCT Frequency Masks (Figure 1)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"📊 DCT masks visualization saved to: {save_path}")
