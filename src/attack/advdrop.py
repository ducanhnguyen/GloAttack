import torch
from advdrop.compression import rgb_to_ycbcr_jpeg, chroma_subsampling, block_splitting, dct_8x8, quantize
from advdrop.decompression import dequantize, idct_8x8, block_merging, chroma_upsampling, ycbcr_to_rgb_jpeg
from src.myutils import (
    load_image_and_targets, exportImage, compute_map_per_image
)


def advdrop(model, sample_img_ids, q_size=10, max_iters=10, alpha_init=0.5, alpha_decay=0.95,
            csv_writer=None, save_dir="out"):
    """
    AdvDrop attack với gradient flow đúng và interface đa hình
    """
    success = 0
    print(f"🎯 Starting AdvDrop attack with q_size={q_size}, max_iters={max_iters}")

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

            # === AdvDrop Attack ===
            attack_successful = False
            final_iteration = 0

            # Prepare frequency domain components
            comps = prepare_ycbcr_blocks(img_tensor)
            q_tables = {}
            dct_blocks = {}

            for k in comps:
                blk = block_splitting(comps[k])  # (B, N, 8, 8)
                dcted = dct_8x8(blk)  # (B, N, 8, 8)
                dct_blocks[k] = dcted
                q_tables[k] = torch.full_like(dcted, q_size, requires_grad=True)

            optimizer = torch.optim.Adam(q_tables.values(), lr=0.01)
            alpha = torch.tensor(alpha_init).to(img_tensor.device)

            for step in range(max_iters):
                optimizer.zero_grad()

                # === Reconstruct image from frequency domain ===
                recons = {}
                for k in comps:
                    quant = quantize(dct_blocks[k], q_tables[k], alpha)
                    dequant = dequantize(quant, q_tables[k])
                    idcted = idct_8x8(dequant)
                    recons[k] = block_merging(idcted, img_tensor.shape[2], img_tensor.shape[3])

                rgb = chroma_upsampling(recons['y'], recons['cb'], recons['cr'])
                adv_tensor = ycbcr_to_rgb_jpeg(rgb)
                adv_tensor = torch.clamp(adv_tensor / 255.0, 0, 1)

                # === Compute loss using polymorphic interface ===
                loss = model.compute_loss(adv_tensor, targets)

                if loss is None or not isinstance(loss, torch.Tensor):
                    print(f"⚠️ Skipping iteration {step} for image {img_id}: Invalid loss")
                    break

                loss.backward(retain_graph=True)

                delta = (adv_tensor - img_tensor).abs().mean().item()
                print(f"📊 Iter {step + 1}/{max_iters}: loss={loss.item():.4f}, alpha={alpha.item():.4f}, diff={delta:.6f}")

                optimizer.step()
                alpha *= alpha_decay

                # === CHECK SUCCESS AFTER EACH ITERATION ===
                pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_tensor)
                map_adv = compute_map_per_image(
                    pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                    gt_boxes, gt_classes
                )

                print(f"📊 Iter {step + 1}/{max_iters}: mAP {map_orig:.4f} → {map_adv:.4f}")

                # Check if attack successful
                if map_adv < map_orig:
                    attack_successful = True
                    final_iteration = step + 1
                    print(f"✅ Attack successful at iteration {final_iteration}!")
                    break

            # === FINAL EVALUATION AND EXPORT ===
            if attack_successful:
                # Get final predictions for export
                pred_boxes_adv, pred_scores_adv, pred_labels_adv = model.predict(adv_tensor)
                map_adv = compute_map_per_image(
                    pred_boxes_adv, pred_scores_adv, pred_labels_adv,
                    gt_boxes, gt_classes
                )

                variant_name = f"advdrop_q{q_size}_iter{final_iteration}"
                exportImage(ori_tensor, gt_classes, adv_tensor, model,
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
    print(f"📊 AdvDrop {model.__class__.__name__}: "
          f"Successful attacks: {success}/{len(sample_img_ids)} ({success_rate:.2f}%)")
    return success


@torch.no_grad()
def prepare_ycbcr_blocks(img_tensor):
    """Tách ảnh RGB thành block DCT từng kênh"""
    imgs = img_tensor.clone() * 255.0
    ycbcr = rgb_to_ycbcr_jpeg(imgs)
    y, cb, cr = chroma_subsampling(ycbcr)
    return {'y': y, 'cb': cb, 'cr': cr}
