import torch
from src.myutils import (
    load_image_and_targets, exportImage, compute_map_per_image
)


def fgsm(model, sample_img_ids, epsilon, csv_writer=None, save_dir="out"):
    """
    FGSM
    """
    success = 0
    model_name = model.__class__.__name__
    print(f"🎯 Starting FGSM attack on {model_name} with epsilon={epsilon}")

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

            # Clone tensor for gradient computation
            img_tensor_grad = img_tensor.clone().detach().requires_grad_(True)

            # === MODEL-SPECIFIC LOSS COMPUTATION ===
            try:
                # Set model to training mode for loss computation
                if hasattr(model, 'model'):
                    model.model.train()
                else:
                    model.train()

                loss = model.compute_loss(img_tensor_grad, targets)

                if loss is None or not isinstance(loss, torch.Tensor):
                    print(f"Skipping image {img_id}: Invalid loss")
                    continue

                # Ensure loss requires gradient
                if not loss.requires_grad:
                    print(f"Loss doesn't require grad for image {img_id}")
                    continue

                # Backward pass
                loss.backward()

                # Get gradient
                grad = img_tensor_grad.grad
                if grad is None:
                    print(f"No gradient computed for image {img_id}")
                    continue

            except Exception as e:
                print(f"Error computing gradient for image {img_id}: {e}")
                continue

            # === CREATE ADVERSARIAL EXAMPLE ===
            adv_tensor = img_tensor + epsilon * grad.sign()
            adv_tensor = torch.clamp(adv_tensor, 0, 1)

            # === EVALUATION ===
            # Set model back to eval mode
            if hasattr(model, 'model'):
                model.model.eval()
            else:
                model.eval()

            # Evaluate original image
            pred_boxes_orig, pred_scores_orig, pred_labels_orig = model.predict(ori_tensor)
            map_orig = compute_map_per_image(
                pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                gt_boxes, gt_classes
            )

            # Evaluate adversarial image
            pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_tensor)
            map_adv = compute_map_per_image(
                pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                gt_boxes, gt_classes
            )

            print(f"📊 Image {img_id}: mAP {map_orig:.4f} → {map_adv:.4f}")

            # === EXPORT RESULTS ===
            if map_adv < map_orig:
                variant_name = f"{model_name}_fgsm_eps{epsilon}_iter{idx}"

                # Handle potential index out of range in exportImage
                try:
                    exportImage(ori_tensor, gt_classes, adv_tensor, model,
                                gt_boxes, img_id, save_dir, map_orig, map_adv,
                                pred_boxes_orig, pred_scores_orig, pred_labels_orig,
                                pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                                variant_name=variant_name, csv_writer=csv_writer)
                    success += 1
                    print(f"Attack successful on image {img_id}")
                except IndexError as ie:
                    print(f"Export error for image {img_id}: {ie}")
                    print(f"   Pred labels orig: {pred_labels_orig}")
                    print(f"   Pred labels adv: {pred_labels_adv}")
                    print(f"   Model classes: {len(model.names) if hasattr(model, 'names') else 'N/A'}")
                    # Still count as success but don't export
                    success += 1
            else:
                print(f"Attack failed on image {img_id}")

        except Exception as e:
            print(f"Error attacking image {img_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # === FINAL STATISTICS ===
    success_rate = (success / len(sample_img_ids)) * 100 if len(sample_img_ids) > 0 else 0
    print(f"📊 FGSM {model_name}: "
          f"Successful attacks: {success}/{len(sample_img_ids)} ({success_rate:.2f}%)")
    return success
