# GloAttack — bản đồ dự án

Repo thí nghiệm paper *Generating adversarial examples based on global frequency domain to attack object detectors*. Tấn công detector trên ảnh COCO val2017, so GloAttack (FFT toàn cục) với baseline: FGSM, PGD, DCT-mask, NumbOD.

Chạy từ **thư mục gốc repo** (`python src/main.py`). Ảnh COCO lấy qua HTTP; annotation nằm local.

---

## Cây thư mục

```
GloAttack/
├── command.sh              # setup: COCO annotations, clone YOLOv5, pip
├── requirements.txt
├── README.md               # hướng dẫn chạy (một số path đã lệch so với code)
├── src/                    # toàn bộ code thí nghiệm
│   ├── main.py             # entry: chọn ảnh → lần lượt các attack → CSV
│   ├── myutils.py          # load ảnh, mAP/IoU, SSIM/PSNR/LPIPS/..., export
│   ├── BaseDetectionModel.py
│   ├── ModelFactory.py     # factory cũ (family-level); thực tế main dùng registry
│   ├── models/
│   │   ├── model_registry.py   # MODEL_TYPE → wrapper (dùng thật)
│   │   ├── YOLOv5Model.py
│   │   ├── FasterRCNNModel.py
│   │   └── DETRModel.py
│   └── attack/
│       ├── myconfig.py     # MODEL_TYPE, NUM_IMAGE, SAVE_DIR, flag export
│       ├── gloattack.py    # phương pháp đề xuất (FFT2)
│       ├── fgsm.py
│       ├── pgd.py
│       ├── gradient_based_dct4od.py   # dct_mask: low / mid / high
│       ├── numod.py        # NumbOD (baseline paper)
│       └── advdrop.py      # có code, main hiện không gọi
├── cache/                  # ID ảnh đã lọc theo model (mAP đủ lớn)
├── result/                 # kết quả thí nghiệm đã chạy (rq1, rq2)
├── annotations/            # tạo bởi command.sh → instances_val2017.json
└── yolov5/                 # clone Ultralytics (YOLOv5Model import từ đây)
```

`out/` được tạo lúc chạy (`SAVE_DIR`). `advdrop/` (JPEG DCT helpers) được `advdrop.py` import nhưng không nằm trong tree source chính.

---

## Luồng chạy (đọc cái này trước)

```
myconfig.py          MODEL_TYPE, NUM_IMAGE, SAVE_DIR, ...
        │
        ▼
main.py
  1. COCO annotations + download ảnh val2017
  2. build_model(MODEL_TYPE)
  3. cache/{MODEL_TYPE}.txt  → nếu thiếu: lọc ảnh mAP > ngưỡng
  4. Lần lượt (mỗi khối `if True:` trong main):
        gloattack → fgsm → pgd → dct_mask → numod
  5. Mỗi cấu hình: thư mục SAVE_DIR/{model}_{attack}_eps..._iter...
        _<tên_thư_mục>.csv + done.txt (skip lần sau)
```

Adapter detector (`compute_loss` / `predict` / `preprocess_input`) nằm trong `src/models/*`. Attack không biết Faster R-CNN vs YOLO vs DETR ngoài interface đó.

**GloAttack (ý tưởng):** FFT2 ảnh → tối ưu nhiễu trên phổ tần số (sign of gradient × epsilon) → IFFT về RGB, clip [0, 1]. Không bound L∞ trên pixel như FGSM/PGD.

---

## File nên mở khi làm việc

| Mục đích | File |
|----------|------|
| Đổi model, số ảnh, thư mục output | `src/attack/myconfig.py` |
| Bật/tắt từng loại tấn công | `src/main.py` (các `if True:` khoảng dòng 366+) |
| Thêm detector | `src/models/<New>.py` + `model_registry.py` + hằng trong `myconfig.py` |
| Metric / load COCO | `src/myutils.py` |
| Thuật toán đề xuất | `src/attack/gloattack.py` |

### `MODEL_TYPE` hợp lệ (`myconfig.py` + `model_registry.py`)

- Faster R-CNN: `faster_rcnn_resnet50`, `faster_rcnn_resnet50_v2`, `faster_rcnn_mobilenet`, `faster_rcnn_mobilenet_320`
- YOLOv5: `yolov5n` / `s` / `m` / `l` / `x` (weight `*.pt` cạnh cwd khi chạy)
- DETR (HuggingFace): `detr-resnet-50`, `detr-resnet-101`, `detr-resnet-50-dc5`

---

## Dữ liệu phụ

- **`cache/{model}.txt`**: danh sách image ID đã lọc cho model đó. Lần chạy sau chỉ lấy `NUM_IMAGE` ID đầu.
- **`result/rq1/`**: CSV theo attack (`gloattack/`, `fgsm/`, `pgd/`, `dct_low|mid|high/`, `numod/`) + `1k images/` (danh sách ID ~1000 ảnh).
- **`result/rq2/`**: CSV so sánh (một số file tên `advlogo` — tên cũ).
- CSV cột chính: `image_id`, `variant`, `map_original`, `map_adversarial`, SSIM, PSNR, MS-SSIM, FSIM, LPIPS, DISTS, L0, L2; GloAttack thêm `delta_R/G/B`. Hàng cuối `AVERAGE`.

---

## Setup nhanh

```bash
bash command.sh          # annotation COCO, clone yolov5, pip
python src/main.py       # cwd = gốc repo
```

Cần: `annotations/annotations/instances_val2017.json`, GPU nếu muốn chạy đúng như thí nghiệm.

---

## Lệch code (để khỏi mất thời gian)

1. `main.py` import `GLOATTACK_CONFIGS`; `myconfig.py` đang khai báo `ADVLOGO_CONFIGS`. Cần cùng một tên trước khi chạy GloAttack.
2. README nói sửa `main.py` / `myconfig.py` ở root; config thật là `src/attack/myconfig.py`.
3. Tất cả attack trong `main.py` đang `if True` — tắt cái không cần bằng `False` kẻo chạy hết baseline.
4. `ModelFactory.py` gần như không dùng; đường vào model là `build_model` trong `model_registry.py`.
5. FGSM/PGD ghép `iter{max_iters}` vào tên thư mục từ vòng GloAttack phía trên (biến leftover), không phải số iter thật của FGSM/PGD.
