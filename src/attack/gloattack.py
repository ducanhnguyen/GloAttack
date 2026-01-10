# src/attack/gloattack.py
import json
import os
import pickle

import torch

import matplotlib.pyplot as plt
import numpy as np
import math
from src.attack.myconfig import ZERO_ATTACK

from src.myutils import (
    load_image_and_targets, exportImage, compute_map_per_image, is_prediction_correct
)


def gloattack(model,
              sample_img_ids,
              epsilon=0.1,
              max_iters=10,
              interval=1,
              visualize_frequency=False,
              csv_writer=None,
              save_dir="out",
              exportOriFolder=False,
              exportAdvFolder=False,
              exportFrequencyNoiseData = False
              ):
    ignore_path = os.path.join(save_dir, "ignore.txt")


    success = 0
    print(f"🎯 Starting GloAttack attack with epsilon={epsilon}, max_iters={max_iters}")

    # 🆕 Tạo thư mục lưu noise data
    if exportFrequencyNoiseData:
        noise_dir = os.path.join(save_dir, "frequency_noise_data")
        os.makedirs(noise_dir, exist_ok=True)


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

            # === FORCE SINGLE IMAGE (CRITICAL INVARIANT) ===
            if img_tensor.dim() == 4 and img_tensor.size(0) > 1:
                img_tensor = img_tensor[:1]

            if ori_tensor.dim() == 4 and ori_tensor.size(0) > 1:
                ori_tensor = ori_tensor[:1]


            baseline_tensor = ori_tensor

            pred_boxes_orig, pred_scores_orig, pred_labels_orig = model.predict(baseline_tensor)
            map_orig = compute_map_per_image(
                pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                gt_boxes, gt_classes
            )

            if map_orig == 0.0:
                print(f"⚠️ Skipping image {img_id}: mAP after defense = 0")

                # === LOG IGNORE IMAGE ID ===
                with open(ignore_path, "a") as f:
                    f.write(f"{img_id}\n")

                continue

            # === Attack ===
            Xf = torch.fft.fft2(baseline_tensor)

            Xf_shifted_original = torch.fft.fftshift(Xf).detach().clone()
            Xf_shifted = Xf_shifted_original.clone().requires_grad_(True)

            all_iterations_data = []
            img_height, img_width = ori_tensor.shape[-2:]
            frequency_noise_data = {
                'img_id': img_id,
                'epsilon': epsilon,
                'max_iters': max_iters,
                'width': img_width,
                'height': img_height,
                'map_orig': map_orig,
                'iterations': []
            }

            original_freq_domain = Xf_shifted_original.detach().clone()
            original_magnitude = torch.abs(original_freq_domain)
            original_phase = torch.angle(original_freq_domain)

            attack_successful = False
            final_iteration = 0

            for step in range(max_iters):
                Xf_shifted_before = Xf_shifted.detach().clone()

                # === IFFT → RGB
                x_recon = torch.fft.ifft2(torch.fft.ifftshift(Xf_shifted)).real
                x_recon = torch.clamp(x_recon, 0, 1)

                # === Compute loss và backward
                loss = model.compute_loss(x_recon, targets)

                if loss is None or not isinstance(loss, torch.Tensor):
                    print(f"⚠️ Skipping iteration {step} for image {img_id}: Invalid loss")
                    break

                loss.backward()

                grad = Xf_shifted.grad.detach()
                direction = grad / (grad.abs() + 1e-12)

                Xf_shifted = (
                        Xf_shifted + epsilon * direction
                ).detach().clone().requires_grad_(True)

                # === Tạo ảnh adversarial
                x_adv = torch.fft.ifft2(torch.fft.ifftshift(Xf_shifted)).real
                x_adv = torch.clamp(x_adv, 0, 1)

                # 🆕 Thu thập dữ liệu nhiễu tần số cho iteration này
                current_freq_domain = Xf_shifted.detach().clone()
                current_magnitude = torch.abs(current_freq_domain)
                current_phase = torch.angle(current_freq_domain)

                # Tính nhiễu theo yêu cầu - cả Sum và L2 distance
                # Frequency domain noise
                freq_domain_diff = current_freq_domain - original_freq_domain
                freq_domain_noise_sum = torch.sum(torch.abs(freq_domain_diff)).item()  # Sum of absolute differences
                freq_domain_noise_l2 = torch.norm(freq_domain_diff).item()  # L2: Euclidean distance

                # Magnitude noise
                magnitude_diff = current_magnitude - original_magnitude
                magnitude_noise_sum = torch.sum(torch.abs(magnitude_diff)).item()
                magnitude_noise_l2 = torch.norm(magnitude_diff).item()

                # Phase noise (xử lý wrap-around cho phase)
                phase_diff = current_phase - original_phase
                phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))  # Wrap to [-π, π]
                phase_noise_sum = torch.sum(torch.abs(phase_diff)).item()
                phase_noise_l2 = torch.norm(phase_diff).item()

                # 🆕 LFC và HFC noise (Low/High Frequency Components)
                # Tạo masks để tách LFC và HFC từ magnitude
                h, w = current_magnitude.shape[-2:]
                center_h, center_w = h // 2, w // 2
                y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                y, x = y.to(current_magnitude.device), x.to(current_magnitude.device)
                distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
                max_distance = min(center_h, center_w)
                threshold_distance = 0.3 * max_distance  # Sử dụng ngưỡng 0.3 như đã thảo luận

                # Low Frequency Components (LFC) - gần tâm
                lfc_mask = distance <= threshold_distance
                lfc_magnitude_diff = magnitude_diff * lfc_mask.float()
                lfc_noise_sum = torch.sum(torch.abs(lfc_magnitude_diff)).item()
                lfc_noise_l2 = torch.norm(lfc_magnitude_diff).item()

                # High Frequency Components (HFC) - xa tâm
                hfc_mask = distance > threshold_distance
                hfc_magnitude_diff = magnitude_diff * hfc_mask.float()
                hfc_noise_sum = torch.sum(torch.abs(hfc_magnitude_diff)).item()
                hfc_noise_l2 = torch.norm(hfc_magnitude_diff).item()

                iteration_noise_data = {
                    'step': step,
                    'freq_domain_noise_sum': freq_domain_noise_sum,
                    'freq_domain_noise_l2': freq_domain_noise_l2,
                    'magnitude_noise_sum': magnitude_noise_sum,
                    'magnitude_noise_l2': magnitude_noise_l2,
                    'phase_noise_sum': phase_noise_sum,
                    'phase_noise_l2': phase_noise_l2,
                    'lfc_noise_sum': lfc_noise_sum,
                    'lfc_noise_l2': lfc_noise_l2,
                    'hfc_noise_sum': hfc_noise_sum,
                    'hfc_noise_l2': hfc_noise_l2,
                    'map_adv': None  # Sẽ được cập nhật sau
                }
                frequency_noise_data['iterations'].append(iteration_noise_data)

                if (step + 1) % interval == 0 or step == max_iters - 1:
                    x_adv_quantized = torch.round(x_adv * 255) / 255

                    pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(x_adv_quantized)

                    map_adv = compute_map_per_image(
                        pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                        gt_boxes, gt_classes
                    )

                    frequency_noise_data['iterations'][-1]['map_adv'] = map_adv

                    # === Thu thập dữ liệu cho evolution visualization
                    if visualize_frequency:
                        iteration_data = collect_iteration_data(
                            step, Xf_shifted_before, Xf_shifted, x_adv,
                            pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                            pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                            map_orig, map_adv
                        )
                        all_iterations_data.append(iteration_data)

                    print(f"📊 Iter {step + 1}/{max_iters}: mAP {map_orig:.4f} → {map_adv:.4f}")

                    if ZERO_ATTACK:
                        if map_adv == 0:
                            attack_successful = True
                            final_iteration = step + 1

                            frequency_noise_data['map_adv_final'] = map_adv
                            frequency_noise_data['final_iteration'] = final_iteration
                            print(f"✅ Attack successful at iteration {final_iteration}!")
                            break

                    elif map_adv < map_orig:
                        attack_successful = True
                        final_iteration = step + 1

                        frequency_noise_data['map_adv_final'] = map_adv
                        frequency_noise_data['final_iteration'] = final_iteration
                        print(f"✅ Attack successful at iteration {final_iteration}!")
                        break

            # === FINAL EVALUATION AND EXPORT ===
            if attack_successful:
                variant_name = f"gloattack_eps{epsilon}_iter{final_iteration}"
                x_adv_quantized = torch.round(x_adv * 255) / 255
                exportImage(ori_tensor, gt_classes, x_adv_quantized, model,
                            gt_boxes, img_id, save_dir, map_orig, map_adv,
                            pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                            pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                            variant_name=variant_name, csv_writer=csv_writer,
                            exportOriFolder=exportOriFolder,
                            exportAdvFolder=exportAdvFolder,
                            )
                success += 1
                print(f"✅ Attack successful on image {img_id} after {final_iteration} iterations")

                if exportFrequencyNoiseData:
                    save_frequency_noise_data(frequency_noise_data, noise_dir)
                    print(f"💾 Frequency noise data saved for image {img_id}")

                    if visualize_frequency and all_iterations_data:
                        visualize_frequency_evolution(
                            ori_tensor, model, gt_boxes, gt_classes, img_id, save_dir,
                            all_iterations_data
                        )
            else:
                print(f"❌ Attack failed on image {img_id} after {max_iters} iterations")

        except Exception as e:
            print(f"❌ Error attacking image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    success_rate = (success / len(sample_img_ids)) * 100 if len(sample_img_ids) > 0 else 0
    print(f"📊 GloAttack {model.__class__.__name__}: "
          f"Successful attacks: {success}/{len(sample_img_ids)} ({success_rate:.2f}%)")
    return success

def save_frequency_noise_data(frequency_noise_data, noise_dir):
    img_id = frequency_noise_data['img_id']

    # Lưu dưới dạng JSON cho dễ đọc
    json_path = os.path.join(noise_dir, f"noise_img_{img_id}.json")
    with open(json_path, 'w') as f:
        json.dump(frequency_noise_data, f, indent=2)

    pickle_path = os.path.join(noise_dir, f"noise_img_{img_id}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(frequency_noise_data, f)

def save_frequency_noise_data(frequency_noise_data, noise_dir):
    """
    🆕 Lưu dữ liệu nhiễu tần số của một ảnh thành công

    Args:
        frequency_noise_data: Dictionary chứa dữ liệu nhiễu của ảnh
        noise_dir: Thư mục lưu dữ liệu
    """
    img_id = frequency_noise_data['img_id']

    json_path = os.path.join(noise_dir, f"noise_img_{img_id}.json")
    with open(json_path, 'w') as f:
        json.dump(frequency_noise_data, f, indent=2)

    pickle_path = os.path.join(noise_dir, f"noise_img_{img_id}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(frequency_noise_data, f)


def create_frequency_noise_summary(noise_dir, num_successful, epsilon, max_iters):
    print(f"\n📊 Creating frequency noise summary for {num_successful} successful attacks...")

    # Thu thập tất cả file json
    json_files = [f for f in os.listdir(noise_dir) if f.endswith('.json')]

    if not json_files:
        print("⚠️ No frequency noise data found")
        return

    summary_data = {
        'experiment_config': {
            'epsilon': epsilon,
            'max_iters': max_iters,
            'num_successful_attacks': num_successful
        },
        'images': [],
        'aggregate_statistics': {}
    }

    all_magnitude_noise = []
    all_phase_noise = []
    all_freq_noise = []
    final_iterations = []

    for json_file in json_files:
        json_path = os.path.join(noise_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_summary = {
            'img_id': data['img_id'],
            'final_iteration': data['final_iteration'],
            'map_orig': data['map_orig'],
            'map_adv_final': data['map_adv_final'],
            'num_iterations': len(data['iterations'])
        }

        # Thu thập statistics từ iteration cuối cùng
        final_iter = data['iterations'][-1]
        img_summary['final_noise_stats'] = final_iter['statistics']

        summary_data['images'].append(img_summary)

        # Thu thập dữ liệu để tính aggregate statistics
        final_iterations.append(data['final_iteration'])
        all_magnitude_noise.append(final_iter['statistics']['magnitude_noise']['l2_norm'])
        all_phase_noise.append(final_iter['statistics']['phase_noise']['l2_norm'])
        all_freq_noise.append(final_iter['statistics']['magnitude']['l2_norm'])

    # Tính aggregate statistics
    summary_data['aggregate_statistics'] = {
        'final_iterations': {
            'mean': np.mean(final_iterations),
            'std': np.std(final_iterations),
            'min': np.min(final_iterations),
            'max': np.max(final_iterations)
        },
        'magnitude_noise_l2': {
            'mean': np.mean(all_magnitude_noise),
            'std': np.std(all_magnitude_noise),
            'min': np.min(all_magnitude_noise),
            'max': np.max(all_magnitude_noise)
        },
        'phase_noise_l2': {
            'mean': np.mean(all_phase_noise),
            'std': np.std(all_phase_noise),
            'min': np.min(all_phase_noise),
            'max': np.max(all_phase_noise)
        },
        'freq_noise_l2': {
            'mean': np.mean(all_freq_noise),
            'std': np.std(all_freq_noise),
            'min': np.min(all_freq_noise),
            'max': np.max(all_freq_noise)
        }
    }

    summary_path = os.path.join(noise_dir, "frequency_noise_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"📄 Frequency noise summary saved: {summary_path}")
    print(f"   📍 Average final iteration: {summary_data['aggregate_statistics']['final_iterations']['mean']:.1f}")
    print(f"   📍 Average magnitude noise L2: {summary_data['aggregate_statistics']['magnitude_noise_l2']['mean']:.4f}")
    print(f"   📍 Average phase noise L2: {summary_data['aggregate_statistics']['phase_noise_l2']['mean']:.4f}")


def load_frequency_noise_data(noise_dir, img_id):
    pickle_path = os.path.join(noise_dir, f"noise_img_{img_id}.pkl")
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    json_path = os.path.join(noise_dir, f"noise_img_{img_id}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    return None


def analyze_frequency_noise_evolution(noise_dir, img_id):
    data = load_frequency_noise_data(noise_dir, img_id)
    if data is None:
        print(f"⚠️ No frequency noise data found for image {img_id}")
        return

    print(f"📊 Frequency Noise Evolution Analysis for Image {img_id}")
    print(f"   📍 Epsilon: {data['epsilon']}")
    print(f"   📍 Final iteration: {data['final_iteration']}")
    print(f"   📍 mAP: {data['map_orig']:.4f} → {data['map_adv_final']:.4f}")
    print(f"   📍 Total iterations analyzed: {len(data['iterations'])}")

    iterations = data['iterations']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    steps = [iter_data['step'] for iter_data in iterations]

    # Plot 1: Magnitude noise evolution
    mag_noise_l2 = [iter_data['statistics']['magnitude']['l2_norm'] for iter_data in iterations]
    axes[0, 0].plot(steps, mag_noise_l2, 'b-o', markersize=4)
    axes[0, 0].set_title('Frequency Magnitude Noise L2 Evolution')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('L2 Norm')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Magnitude component noise evolution
    mag_comp_l2 = [iter_data['statistics']['magnitude_noise']['l2_norm'] for iter_data in iterations]
    axes[0, 1].plot(steps, mag_comp_l2, 'g-o', markersize=4)
    axes[0, 1].set_title('Magnitude Component Noise L2 Evolution')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('L2 Norm')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Phase noise evolution
    phase_noise_l2 = [iter_data['statistics']['phase_noise']['l2_norm'] for iter_data in iterations]
    axes[1, 0].plot(steps, phase_noise_l2, 'r-o', markersize=4)
    axes[1, 0].set_title('Phase Noise L2 Evolution')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: mAP evolution (nếu có)
    map_values = [iter_data.get('map_adv', None) for iter_data in iterations]
    map_values = [val for val in map_values if val is not None]
    map_steps = steps[-len(map_values):] if map_values else []

    if map_values:
        axes[1, 1].plot(map_steps, map_values, 'm-o', markersize=4)
        axes[1, 1].axhline(y=data['map_orig'], color='k', linestyle='--', alpha=0.7, label='Original mAP')
        axes[1, 1].set_title('mAP Evolution')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('mAP')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No mAP data available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('mAP Evolution (No Data)')

    plt.suptitle(f'Frequency Noise Evolution - Image {img_id}', fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(noise_dir, f"noise_evolution_img_{img_id}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Evolution plot saved: {plot_path}")


def plot_gam_grid(gam_list, save_path, cmap="hot"):
    num = len(gam_list)
    cols = min(5, num)
    rows = math.ceil(num / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))

    axs = np.array(axs).reshape(rows, cols)

    vmin = min([g.min() for g in gam_list])
    vmax = max([g.max() for g in gam_list])

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axs[r][c]
        ax.axis("off")
        if i < num:
            im = ax.imshow(gam_list[i], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Iter {i + 1}", fontsize=12, pad=8)

    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Gradient Magnitude', fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # chừa chỗ cho colorbar bên dưới
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_frequency_evolution(ori_tensor, model, gt_boxes, gt_classes, img_id, save_dir,
                                  all_iterations_data):

    def to_numpy_freq(tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.permute(1, 2, 0).cpu().detach().numpy()

    def to_numpy_image(tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        return tensor.permute(1, 2, 0).cpu().detach().numpy()

    def norm_for_display(x):
        x_min, x_max = x.min(), x.max()

        # Handle edge case: all values are the same
        if x_max - x_min < 1e-8:
            return np.zeros_like(x)

        # Simple min-max normalization
        normalized = (x - x_min) / (x_max - x_min)

        # Ensure exactly [0,1] range
        return np.clip(normalized, 0, 1)

    def create_frequency_mask(freq_tensor, mask_type="high", threshold_ratio=0.3):
        h, w = freq_tensor.shape[-2:]
        center_h, center_w = h // 2, w // 2

        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.to(freq_tensor.device), x.to(freq_tensor.device)

        distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
        max_distance = min(center_h, center_w)
        threshold_distance = threshold_ratio * max_distance

        if mask_type == "high":
            mask = distance > threshold_distance  # High frequency
        else:  # low
            mask = distance <= threshold_distance  # Low frequency

        return mask

    if not all_iterations_data:
        return

    num_iters = len(all_iterations_data)
    final_data = all_iterations_data[-1]

    Xf_original = torch.fft.fft2(ori_tensor)
    Xf_shifted_original = torch.fft.fftshift(Xf_original)

    freq_magnitude = torch.abs(Xf_shifted_original)
    freq_log_magnitude = torch.log1p(freq_magnitude)
    freq_amplitude = freq_log_magnitude  # Phổ biên độ (có log)
    freq_phase = torch.angle(Xf_shifted_original)

    ori_np = to_numpy_image(ori_tensor)
    adv_final_np = to_numpy_image(final_data['x_adv'])
    freq_magnitude_np = to_numpy_freq(freq_magnitude)
    freq_log_np = to_numpy_freq(freq_log_magnitude)
    freq_amplitude_np = to_numpy_freq(freq_amplitude)
    freq_phase_np = to_numpy_freq(freq_phase)

    # Row 2-7
    freq_noise_evolution = []
    amplitude_noise_evolution = []
    phase_noise_evolution = []
    high_freq_noise_evolution = []
    low_freq_noise_evolution = []
    image_noise_evolution = []

    Xf_shifted_initial = all_iterations_data[0]['Xf_shifted_before']
    initial_amplitude = torch.log1p(torch.abs(Xf_shifted_initial))
    initial_phase = torch.angle(Xf_shifted_initial)

    for i, data in enumerate(all_iterations_data):
        #
        freq_noise = torch.abs(data['Xf_shifted_after'] - Xf_shifted_initial)
        freq_noise_evolution.append(to_numpy_freq(freq_noise))

        #
        current_amplitude = torch.log1p(torch.abs(data['Xf_shifted_after']))
        amplitude_noise = torch.abs(current_amplitude - initial_amplitude)
        amplitude_noise_evolution.append(to_numpy_freq(amplitude_noise))

        #
        current_phase = torch.angle(data['Xf_shifted_after'])
        phase_noise = torch.abs(torch.atan2(torch.sin(current_phase - initial_phase),
                                            torch.cos(current_phase - initial_phase)))
        phase_noise_evolution.append(to_numpy_freq(phase_noise))

        #
        high_freq_mask = create_frequency_mask(data['Xf_shifted_after'], "high", 0.3)
        freq_diff = data['Xf_shifted_after'] - Xf_shifted_initial
        high_freq_noise = torch.abs(freq_diff) * high_freq_mask.float()
        high_freq_noise_evolution.append(to_numpy_freq(high_freq_noise))

        #
        low_freq_mask = create_frequency_mask(data['Xf_shifted_after'], "low", 0.3)
        low_freq_noise = torch.abs(freq_diff) * low_freq_mask.float()
        low_freq_noise_evolution.append(to_numpy_freq(low_freq_noise))

        #
        image_noise = torch.abs(data['x_adv'] - ori_tensor)
        image_noise_evolution.append(to_numpy_image(image_noise))

    max_cols = max(6, num_iters)
    fig, axs = plt.subplots(7, max_cols, figsize=(4 * max_cols, 28))
    font_size = 10

    if max_cols == 1:
        axs = axs.reshape(7, 1)

    axs[0, 0].imshow(ori_np)
    axs[0, 0].set_title(f"Original image\n(mAP {final_data['map_orig']:.3f})", fontsize=14)
    axs[0, 0].axis("off")

    for box, score, cls in zip(final_data['pred_boxes_orig'], final_data['pred_scores_orig'],
                               final_data['pred_labels_orig']):
        x1, y1, x2, y2 = box
        label = model.names[cls] if hasattr(model, 'names') else str(cls)
        color = 'green' if is_prediction_correct(box, cls, gt_boxes, gt_classes) else 'red'
        axs[0, 0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          edgecolor=color, linestyle='--', fill=False, linewidth=2))
        axs[0, 0].text(x1, y1 - 5, f"{label}\n{score:.2f}", fontsize=font_size,
                       color=color, backgroundcolor='white', verticalalignment='top')

    #
    axs[0, 1].imshow(norm_for_display(freq_magnitude_np), cmap="gray")
    axs[0, 1].set_title("Miền tần số\n(chưa log)", fontsize=14)
    axs[0, 1].axis("off")

    #
    axs[0, 2].imshow(norm_for_display(freq_log_np), cmap="gray")
    axs[0, 2].set_title("Miền tần số\n(có log)", fontsize=14)
    axs[0, 2].axis("off")

    #
    axs[0, 3].imshow(norm_for_display(freq_amplitude_np), cmap="hot")
    axs[0, 3].set_title("Phổ biên độ\n(có log)", fontsize=14)
    axs[0, 3].axis("off")

    #
    axs[0, 4].imshow(norm_for_display(freq_phase_np), cmap="hsv")
    axs[0, 4].set_title("Phổ pha", fontsize=14)
    axs[0, 4].axis("off")

    #
    axs[0, 5].imshow(adv_final_np)
    axs[0, 5].set_title(f"Ảnh adversarial\n(mAP {final_data['map_adv']:.3f})", fontsize=14)
    axs[0, 5].axis("off")

    #
    for box, score, cls in zip(final_data['pred_boxes_adv'], final_data['pred_scores_adv'],
                               final_data['pred_labels_adv']):
        x1, y1, x2, y2 = box
        label = model.names[cls] if hasattr(model, 'names') else str(cls)
        color = 'green' if is_prediction_correct(box, cls, gt_boxes, gt_classes) else 'red'
        axs[0, 5].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          edgecolor=color, linestyle='--', fill=False, linewidth=2))
        axs[0, 5].text(x1, y1 - 5, f"{label}\n{score:.2f}", fontsize=font_size,
                       color=color, backgroundcolor='white', verticalalignment='top')

    #
    for col in range(6, max_cols):
        axs[0, col].axis("off")

    #
    for i in range(num_iters):
        noise_raw = freq_noise_evolution[i]
        noise_norm = norm_for_display(noise_raw)
        im = axs[1, i].imshow(noise_norm, cmap="hot", vmin=0, vmax=1)

        min_val = np.min(noise_raw)
        max_val = np.max(noise_raw)
        axs[1, i].set_title(f"Iter {i + 1}\nFreq noise\nMin: {min_val:.3f}\nMax: {max_val:.3f}", fontsize=16)
        axs[1, i].axis("off")

        #
        if i == 0:
            plt.colorbar(im, ax=axs[1, i], shrink=0.6)

    #
    for col in range(num_iters, max_cols):
        axs[1, col].axis("off")

    # ROW 3
    for i in range(num_iters):
        noise_raw = amplitude_noise_evolution[i]
        noise_norm = norm_for_display(noise_raw)
        im = axs[2, i].imshow(noise_norm, cmap="hot", vmin=0, vmax=1)

        min_val = np.min(noise_raw)
        max_val = np.max(noise_raw)
        axs[2, i].set_title(f"Iter {i + 1}\nAmplitude noise\nMin: {min_val:.3f}\nMax: {max_val:.3f}", fontsize=16)
        axs[2, i].axis("off")

        #
        if i == 0:
            plt.colorbar(im, ax=axs[2, i], shrink=0.6)

    #
    for col in range(num_iters, max_cols):
        axs[2, col].axis("off")

    #
    for i in range(num_iters):
        noise_raw = phase_noise_evolution[i]
        noise_norm = norm_for_display(noise_raw)
        im = axs[3, i].imshow(noise_norm, cmap="hot", vmin=0, vmax=1)

        min_val = np.min(noise_raw)
        max_val = np.max(noise_raw)
        axs[3, i].set_title(f"Iter {i + 1}\nPhase noise\nMin: {min_val:.3f}\nMax: {max_val:.3f}", fontsize=16)
        axs[3, i].axis("off")

        #
        if i == 0:
            plt.colorbar(im, ax=axs[3, i], shrink=0.6)

    #
    for col in range(num_iters, max_cols):
        axs[3, col].axis("off")

    #
    for i in range(num_iters):
        noise_raw = high_freq_noise_evolution[i]
        noise_norm = norm_for_display(noise_raw)
        im = axs[4, i].imshow(noise_norm, cmap="hot", vmin=0, vmax=1)

        min_val = np.min(noise_raw)
        max_val = np.max(noise_raw)
        axs[4, i].set_title(f"Iter {i + 1}\nHigh freq noise\nMin: {min_val:.3f}\nMax: {max_val:.3f}", fontsize=16)
        axs[4, i].axis("off")

        if i == 0:
            plt.colorbar(im, ax=axs[4, i], shrink=0.6)

    for col in range(num_iters, max_cols):
        axs[4, col].axis("off")

    # ROW 6
    for i in range(num_iters):
        noise_raw = low_freq_noise_evolution[i]
        noise_norm = norm_for_display(noise_raw)
        im = axs[5, i].imshow(noise_norm, cmap="hot", vmin=0, vmax=1)

        min_val = np.min(noise_raw)
        max_val = np.max(noise_raw)
        axs[5, i].set_title(f"Iter {i + 1}\nLow freq noise\nMin: {min_val:.3f}\nMax: {max_val:.3f}", fontsize=16)
        axs[5, i].axis("off")

        #
        if i == 0:
            plt.colorbar(im, ax=axs[5, i], shrink=0.6)

    #
    for col in range(num_iters, max_cols):
        axs[5, col].axis("off")

    #
    for i in range(num_iters):
        img_noise = image_noise_evolution[i]
        # Chuyển về grayscale nếu là RGB
        if len(img_noise.shape) == 3:
            img_noise_gray = np.mean(img_noise, axis=2)
        else:
            img_noise_gray = img_noise

        noise_norm = norm_for_display(img_noise_gray)
        im = axs[6, i].imshow(noise_norm, cmap="hot", vmin=0, vmax=1)

        min_val = np.min(img_noise)
        max_val = np.max(img_noise)
        l2_norm = np.linalg.norm(img_noise)
        axs[6, i].set_title(f"Iter {i + 1}\nImage noise\nMin: {min_val:.3f}\nMax: {max_val:.3f}\nL2: {l2_norm:.3f}", fontsize=16)
        axs[6, i].axis("off")

        if i == 0:
            plt.colorbar(im, ax=axs[6, i], shrink=0.6)

    for col in range(num_iters, max_cols):
        axs[6, col].axis("off")

    plt.tight_layout()

    # Save image
    freq_viz_dir = os.path.join(save_dir, "frequency_evolution")
    os.makedirs(freq_viz_dir, exist_ok=True)
    freq_path = os.path.join(freq_viz_dir, f"evolution_{img_id}.png")
    plt.savefig(freq_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📊 Evolution analysis cho ảnh {img_id} đã lưu tại: {freq_path}")
    print(f"    📍 Số iterations: {num_iters}")

    #
    print(f"    📍 High freq noise evolution:")
    for i, noise in enumerate(high_freq_noise_evolution):
        min_noise = np.min(noise)
        max_noise = np.max(noise)
        print(f"      Iter {i + 1}: Min={min_noise:.3f}, Max={max_noise:.3f}")

    print(f"    📍 Low freq noise evolution:")
    for i, noise in enumerate(low_freq_noise_evolution):
        min_noise = np.min(noise)
        max_noise = np.max(noise)
        print(f"      Iter {i + 1}: Min={min_noise:.3f}, Max={max_noise:.3f}")

    print(f"    📍 Image noise evolution:")
    for i, noise in enumerate(image_noise_evolution):
        min_noise = np.min(noise)
        max_noise = np.max(noise)
        l2_noise = np.linalg.norm(noise)
        print(f"      Iter {i + 1}: Min={min_noise:.3f}, Max={max_noise:.3f}, L2={l2_noise:.3f}")

def collect_iteration_data(step, Xf_shifted_before, Xf_shifted_after, x_adv,
                           pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                           pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                           map_orig, map_adv):
    return {
        'step': step,
        'Xf_shifted_before': Xf_shifted_before.detach().clone(),
        'Xf_shifted_after': Xf_shifted_after.detach().clone(),
        'x_adv': x_adv.detach().clone(),
        'pred_boxes_orig': pred_boxes_orig.copy(),
        'pred_scores_orig': pred_scores_orig.copy(),
        'pred_labels_orig': pred_labels_orig.copy(),
        'pred_boxes_adv': pred_boxes_adv.copy(),
        'pred_scores_adv': pred_scores_adv.copy(),
        'pred_labels_adv': pred_labels_adv.copy(),
        'map_orig': map_orig,
        'map_adv': map_adv
    }

