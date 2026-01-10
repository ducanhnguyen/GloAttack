# checked
import torch
from src.myutils import (
    load_image_and_targets, exportImage, compute_map_per_image
)


def pgd(model, sample_img_ids, epsilon=0.03, alpha=0.005, num_iter=10, random_start=True, csv_writer=None,
        save_dir="out"):
    success = 0
    print(f"🎯 Starting PGD attack with epsilon={epsilon}, alpha={alpha}, iter={num_iter}")

    total_images = len(sample_img_ids)
    for idx, img_id in enumerate(sample_img_ids, 1):
        print("=" * 50)
        print(f"🖼️ Image {img_id} [{idx:2d}/{total_images}]")

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

            # === PGD Attack ===
            # 1. Random initialization (nếu random_start=True)
            if random_start:
                # Khởi tạo ngẫu nhiên trong epsilon-ball
                noise = torch.empty_like(ori_tensor).uniform_(-epsilon, epsilon)
                adv_tensor = ori_tensor + noise
                adv_tensor = torch.clamp(adv_tensor, 0, 1)  # Đảm bảo trong [0,1]
            else:
                adv_tensor = ori_tensor.clone()

            # 2. Iterative attacks
            attack_successful = False
            final_iteration = 0

            for i in range(num_iter):
                adv_tensor.requires_grad_(True)

                # Forward pass - compute loss
                loss = model.compute_loss(adv_tensor, targets)

                if loss is None or not isinstance(loss, torch.Tensor):
                    print(f"Skipping iteration {i} for image {img_id}: Invalid loss")
                    break

                # Backward pass
                loss.backward()

                # 3. Update với gradient ascent (để maximize loss)
                grad = adv_tensor.grad.detach()
                adv_tensor = adv_tensor.detach() + alpha * grad.sign()

                # 4. Project về epsilon-ball và [0,1]
                # Đầu tiên project về epsilon-ball
                perturbation = adv_tensor - ori_tensor
                perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                adv_tensor = ori_tensor + perturbation

                # Sau đó clamp về [0,1]
                adv_tensor = torch.clamp(adv_tensor, 0, 1)

                # === CHECK SUCCESS AFTER EACH ITERATION ===
                pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_tensor)
                map_adv = compute_map_per_image(
                    pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                    gt_boxes, gt_classes
                )

                print(f"📊 Iter {i + 1}/{num_iter}: mAP {map_orig:.4f} → {map_adv:.4f}")

                # Check if attack successful
                if map_adv < map_orig:
                    attack_successful = True
                    final_iteration = i + 1
                    print(f"Attack successful at iteration {final_iteration}!")
                    break

            # === FINAL EVALUATION AND EXPORT ===
            if attack_successful:
                # Get final predictions for export
                pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_tensor)
                map_adv = compute_map_per_image(
                    pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                    gt_boxes, gt_classes
                )

                variant_name = f"pgd_eps{epsilon}_alpha{alpha}_iter{final_iteration}"
                exportImage(ori_tensor, gt_classes, adv_tensor, model,
                            gt_boxes, img_id, save_dir, map_orig, map_adv,
                            pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                            pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                            variant_name=variant_name, csv_writer=csv_writer)
                success += 1
                print(f"Attack successful on image {img_id} after {final_iteration} iterations")
            else:
                print(f"Attack failed on image {img_id} after {num_iter} iterations")

        except Exception as e:
            print(f"Error attacking image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # === THỐNG KÊ KẾT QUẢ ===
    success_rate = (success / len(sample_img_ids)) * 100 if len(sample_img_ids) > 0 else 0
    print(f"PGD {model.__class__.__name__}: "
          f"Successful attacks: {success}/{len(sample_img_ids)} ({success_rate:.2f}%)")
    return success