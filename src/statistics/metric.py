"""
src/statistics/metric.py

So sanh phuong phap tan cong de xuat (pp A, mac dinh: "gloattack") voi cac
phuong phap baseline khac (pp B: fgsm, pgd, dct_low, dct_mid, dct_high,
numod, ...) THEO TUNG MODEL BI TAN CONG va THEO TUNG METRIC chat luong anh,
dung Wilcoxon signed-rank test (paired, ghep cap theo image_id).

CACH TO CHUC DU LIEU DAU VAO (gia dinh)
----------------------------------------
BASE_DIR (mac dinh "result/rq1") chua nhieu THU MUC CON, moi thu muc con
la mot PHUONG PHAP (vi du: gloattack/, fgsm/, pgd/, dct_low/, dct_mid/,
dct_high/, numod/), TRU thu muc "1k images" (khong phai phuong phap, bi
bo qua).

Trong moi thu muc phuong phap co NHIEU file .csv, MOI FILE LA KET QUA TAN
CONG CUA MOT MODEL (ten model nam trong ten file, vi du
"..._detr-resnet-50...csv", "..._yolov5l...csv"). Danh sach model hop le
(MODEL_NAMES ben duoi) da duoc co dinh san - CHI CAC MODEL NAY moi duoc dua
vao bang ket qua; file nao khong khop ten model nao trong danh sach se bi
BO QUA va bao canh bao ra man hinh.

Moi file .csv co cot "image_id" va cac cot metric: SSIM, PSNR, MS-SSIM,
FSIM, LPIPS, DISTS, L0, L2 (co the co them map_original, map_adversarial,
variant, ... - khong dung o day). Hang cuoi kieu "AVERAGE" (image_id khong
phai so) se bi loai bo.

GHEP CAP THEO TUNG MODEL
--------------------------
Voi moi (model, pp A, pp B), doc RIENG file cua model do trong thu muc
pp A va file cua model do trong thu muc pp B, roi chi giu lai cac image_id
XUAT HIEN O CA HAI (giao) de ghep cap cho Wilcoxon signed-rank test - khac
voi ban truoc, o day KHONG gop/trung binh nhieu model lai voi nhau, moi
model duoc kiem dinh rieng.

BANG KET QUA (moi hang = 1 model x 1 cap pp x 1 metric)
----------------------------------------------------------
  Cot 1 model         : ten model (yolov5l, detr-resnet-50, ...)
  Cot 2 pp_A           : ten phuong phap de xuat (mac dinh gloattack)
  Cot 3 pp_B           : ten phuong phap doi sanh
  Cot 4 metric         : SSIM / PSNR / MS-SSIM / FSIM / LPIPS / DISTS / L0 / L2
  Cot 5 common_images   : "so image_id trung / tong so image_id (ti le %)",
                          voi "tong" = HOP (union) cua tap image_id o pp A
                          va pp B cho model do; "trung" = GIAO (intersection),
                          chinh la so cap du lieu dung de kiem dinh.
  Cot 6 p_value        : p-value hai phia (two-sided) cua Wilcoxon
                          signed-rank test so sanh pp A vs pp B tren tap
                          image_id trung nhau, cho metric do.

  Cac cot bo sung (them vao SAU 6 cot chinh, de co du lieu chung minh
  "tot hon" thay vi chi nhin p-value tho):
  - metric_direction    : "higher_better" (SSIM/PSNR/MS-SSIM/FSIM - cang
                          cao cang tot) hoac "lower_better" (LPIPS/DISTS/
                          L0/L2 - cang thap cang tot)
  - A_median, B_median  : trung vi metric CUA CHINH TAP ANH TRUNG (khong
                          phai toan bo anh) cho pp A / pp B
  - A_std, B_std        : do lech chuan (ddof=1) tren tap anh trung
  - A_iqr, B_iqr        : khoang tu phan vi (Q75-Q25) tren tap anh trung -
                          do phan tan ben voi outlier hon std, phu hop khi
                          du lieu khong chac phan phoi chuan
  - median_diff         : trung vi (A - B)
  - median_diff_ci_low,
    median_diff_ci_high : khoang tin cay 95% (percentile bootstrap, xem
                          --n-bootstrap) cho median_diff - THUOC DO DO TIN
                          CAY: neu khoang nay rong hoac chua ca gia tri 0,
                          "loi the" cua pp A khong on dinh du p-value nho
  - rank_biserial_r     : he so tuong quan rank-biserial ghep cap, [-1,1]
  - effect_size_label   : dien giai |rank_biserial_r| (negligible/small/
                          medium/large) theo nguong tham khao
  - n_pairs_used        : so cap dung de kiem dinh (da bo cap bang nhau
                          tuyet doi theo quy uoc Wilcoxon "wilcox")
  - wilcoxon_statistic  : thong ke W tra ve tu scipy
  - significant_p<0.05  : True neu p_value (CHUA hieu chinh) < 0.05
  - p_value_fdr         : p-value da hieu chinh Benjamini-Hochberg (FDR)
                          tren TOAN BO cac phep kiem dinh trong bang - vi
                          bang co rat nhieu phep test cung luc (nhieu
                          model x nhieu metric x nhieu cap pp), p_value
                          tho de bi duong tinh gia; NEN DUNG cot nay khi
                          bao cao trong paper thay vi p_value tho
  - significant_fdr_q<0.05 : True neu p_value_fdr < 0.05 (dang tin cay
                          hon significant_p<0.05 khi so sanh nhieu lan)
  - A_better            : True/False CHI KHI co y nghia thong ke (p<0.05,
                          CHUA hieu chinh FDR), ket hop metric_direction +
                          median_diff; NaN neu khong du bang chung de
                          ket luan

Chay tu THU MUC GOC REPO:
    python src/statistics/metric.py
Hoac tuy chinh:
    python src/statistics/metric.py --base-dir result/rq1 \
        --proposed gloattack --output result/metric.csv
    # chi so sanh voi mot vai pp B cu the (thay vi tat ca):
    python src/statistics/metric.py --proposed gloattack --other fgsm pgd
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

# ---------------------------------------------------------------------------
# Cau hinh mac dinh
# ---------------------------------------------------------------------------

DEFAULT_BASE_DIR = "result/rq1"
DEFAULT_PROPOSED_METHOD = "gloattack"
DEFAULT_OUTPUT = "result/metric.csv"

# Thu muc bi loai (khong phai phuong phap thuc nghiem)
EXCLUDED_DIR_NAMES = {"1k images", "1k_images", "1kimages"}

METRIC_COLUMNS = [
    "SSIM",
    "PSNR",
    "MS-SSIM",
    "FSIM",
    "LPIPS",
    "DISTS",
    "L0",
    "L2",
]

ID_COLUMN = "image_id"

# Chieu "tot hon" cua tung metric:
#   higher_better: SSIM/PSNR/MS-SSIM/FSIM cang cao nghia la anh doi khang
#       cang giong anh goc (kho nhan biet hon / chat luong nhieu tot hon)
#   lower_better : LPIPS/DISTS la khoang cach cam nhan, L0/L2 la do lon
#       nhieu (perturbation) -> cang thap nghia la anh doi khang cang giong
#       anh goc / nhieu cang nho -> tot hon
METRIC_DIRECTION = {
    "SSIM": "higher_better",
    "PSNR": "higher_better",
    "MS-SSIM": "higher_better",
    "FSIM": "higher_better",
    "LPIPS": "lower_better",
    "DISTS": "lower_better",
    "L0": "lower_better",
    "L2": "lower_better",
}

ALPHA = 0.05  # nguong y nghia thong ke
N_BOOTSTRAP_DEFAULT = 2000  # so lan resample cho khoang tin cay bootstrap
BOOTSTRAP_SEED = 42  # co dinh de ket qua tai lap duoc (reproducible)

# Danh sach model co san (co dinh, theo dung ten ban cung cap). Thu tu o
# day cung la thu tu xuat hien trong bang ket qua.
MODEL_NAMES = [
    "detr-resnet-50",
    "detr-resnet-50-dc5",
    "detr-resnet-101",
    "faster_rcnn_mobilenet_320",
    "faster_rcnn_resnet50",
    "faster_rcnn_resnet50_v2",
    "yolov5l",
    "yolov5m",
    "yolov5n",
    "yolov5s",
    "yolov5x",
]

# Sap xep giam dan theo do dai de khop ten model trong ten file: tranh
# truong hop "faster_rcnn_resnet50" khop nham vao file cua
# "faster_rcnn_resnet50_v2" (hoac "detr-resnet-50" khop nham vao file cua
# "detr-resnet-50-dc5"). Model dai hon duoc kiem tra truoc.
_MODEL_NAMES_BY_LEN_DESC = sorted(MODEL_NAMES, key=len, reverse=True)


# ---------------------------------------------------------------------------
# Doc du lieu
# ---------------------------------------------------------------------------

def is_excluded_dir(dirname: str) -> bool:
    normalized = dirname.strip().lower()
    return normalized in EXCLUDED_DIR_NAMES


def match_model_name(file_stem: str) -> str | None:
    """Tim ten model (trong MODEL_NAMES) xuat hien trong ten file (khong
    phan biet hoa/thuong). Tra ve None neu khong khop model nao."""
    low = file_stem.lower()
    for model in _MODEL_NAMES_BY_LEN_DESC:
        if model.lower() in low:
            return model
    return None


def index_method_files(method_dir: Path) -> dict[str, list[Path]]:
    """Gan moi file .csv trong thu muc phuong phap vao dung 1 model (dua
    tren ten file). Tra ve dict model -> danh sach file (thuong chi co 1
    file/model, nhung van ho tro nhieu file neu co)."""
    index: dict[str, list[Path]] = {m: [] for m in MODEL_NAMES}
    unmatched: list[str] = []

    for f in sorted(method_dir.glob("*.csv")):
        model = match_model_name(f.stem)
        if model is None:
            unmatched.append(f.name)
        else:
            index[model].append(f)

    n_found = sum(1 for m in MODEL_NAMES if index[m])
    print(f"  {method_dir}: khop duoc {n_found}/{len(MODEL_NAMES)} model")
    for model in MODEL_NAMES:
        files = index[model]
        if not files:
            print(f"    [thieu] khong tim thay file cho model '{model}'")
        elif len(files) > 1:
            print(f"    [canh bao] model '{model}' khop {len(files)} file: "
                  f"{[x.name for x in files]}")
    if unmatched:
        print(f"    [BO QUA] {len(unmatched)} file khong khop ten model nao "
              f"trong MODEL_NAMES: {unmatched}")

    return index


def load_csv_files(csv_files: list[Path], label: str) -> pd.DataFrame:
    """Doc mot danh sach file .csv (thuong la 1 file cua 1 model), gop lai,
    loai hang khong co image_id hop le (vi du hang "AVERAGE"), gom nhom
    theo image_id va lay TRUNG BINH neu image_id lap lai giua cac file.

    Tra ve DataFrame index = image_id (int), cot = cac metric co mat.
    """
    if not csv_files:
        return pd.DataFrame(columns=METRIC_COLUMNS).set_index(pd.Index([], name=ID_COLUMN))

    frames = []
    for f in csv_files:
        # Kiem tra truoc: so cot khai bao trong dong header co KHOP voi so
        # gia tri thuc te trong dong du lieu dau tien khong. Neu LECH (vi
        # du file thieu khai bao cot delta_R/delta_G/delta_B nhung van ghi
        # du lieu 3 cot do), pandas se TU DONG coi cac cot du ra la INDEX
        # va gan nham ten cot cho du lieu bi lech - dan den "image_id" doc
        # ra toan la 0 (hoac gia tri sai) ma KHONG BAO LOI GI CA. De tranh
        # am tham doc sai, ta kiem tra va BO QUA file nay, bao canh bao ro.
        try:
            with open(f, "r", newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.reader(fh)
                header_row = next(reader, None)
                first_data_row = next(reader, None)
        except Exception as exc:  # noqa: BLE001
            print(f"    [BO QUA] {label}: khong doc duoc {f.name} ({exc})")
            continue

        if header_row is None:
            print(f"    [BO QUA] {label}: {f.name} rong (khong co header)")
            continue

        if first_data_row is not None and len(first_data_row) != len(header_row):
            print(
                f"    [BO QUA] {label}: {f.name} bi LECH COT - header co "
                f"{len(header_row)} cot nhung dong du lieu dau tien co "
                f"{len(first_data_row)} gia tri. File nay THIEU KHAI BAO "
                f"COT trong dong header (vi du thieu delta_R/delta_G/"
                f"delta_B) - neu doc binh thuong, pandas se tu dich cot "
                f"va lam SAI toan bo du lieu (ke ca image_id). Can sua lai "
                f"dong header trong file nguon truoc khi chay lai."
            )
            continue

        try:
            df = pd.read_csv(f)
        except Exception as exc:  # noqa: BLE001
            print(f"    [BO QUA] {label}: khong doc duoc {f.name} ({exc})")
            continue

        n_rows_raw = len(df)
        df.columns = [str(c).strip() for c in df.columns]

        if ID_COLUMN not in df.columns:
            print(f"    [BO QUA] {label}: {f.name} khong co cot '{ID_COLUMN}' "
                  f"(cac cot: {list(df.columns)})")
            continue

        present_metrics = [m for m in METRIC_COLUMNS if m in df.columns]
        missing_metrics = [m for m in METRIC_COLUMNS if m not in df.columns]
        if missing_metrics:
            print(f"    [canh bao] {label}: {f.name} thieu cot {missing_metrics}")

        df = df[[ID_COLUMN] + present_metrics].copy()

        df[ID_COLUMN] = pd.to_numeric(df[ID_COLUMN], errors="coerce")
        n_bad_id = int(df[ID_COLUMN].isna().sum())
        df = df.dropna(subset=[ID_COLUMN])
        df[ID_COLUMN] = df[ID_COLUMN].astype(int)

        for m in present_metrics:
            df[m] = pd.to_numeric(df[m], errors="coerce")

        n_kept = len(df)
        warn = f" ({n_bad_id} dong bi loai do image_id khong hop le)" if n_bad_id else ""
        print(f"    {label}: {f.name}: {n_rows_raw} dong -> giu {n_kept} dong{warn}")

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=METRIC_COLUMNS).set_index(pd.Index([], name=ID_COLUMN))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    grouped = combined.groupby(ID_COLUMN, as_index=True).mean(numeric_only=True)
    return grouped


def discover_methods(base_dir: Path) -> dict[str, Path]:
    """Tim cac thu muc con = phuong phap trong base_dir (tru thu muc bi loai)."""
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay thu muc du lieu: {base_dir}")

    methods = {}
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        if is_excluded_dir(entry.name):
            continue
        methods[entry.name] = entry
    return methods


# ---------------------------------------------------------------------------
# Thong ke
# ---------------------------------------------------------------------------

def rank_biserial_from_wilcoxon(diffs: np.ndarray) -> tuple[float, float, float]:
    """
    Tinh W+, W- va he so tuong quan rank-biserial ghep cap cho Wilcoxon
    signed-rank test, dua tren cac hieu so (diffs) DA LOAI gia tri = 0.

    r = (W+ - W-) / (W+ + W-)
    """
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return np.nan, np.nan, np.nan

    abs_ranks = rankdata(np.abs(diffs))
    w_pos = abs_ranks[diffs > 0].sum()
    w_neg = abs_ranks[diffs < 0].sum()
    denom = w_pos + w_neg
    r = (w_pos - w_neg) / denom if denom > 0 else np.nan
    return w_pos, w_neg, r


def bootstrap_median_diff_ci(
    diffs: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Khoang tin cay (percentile bootstrap) cho TRUNG VI cua diffs (A - B).
    Day la thuoc do DO TIN CAY cua "loi the" (median_diff) - neu khoang
    tin cay rong hoac chua ca gia tri 0, ket qua khong on dinh du p-value
    co the < 0.05 (dac biet quan trong voi mau nho).

    Tra ve (ci_low, ci_high). NaN neu khong du du lieu de resample.
    """
    n = len(diffs)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        # Chi 1 cap: khong the resample co y nghia, tra ve chinh gia tri do
        return float(diffs[0]), float(diffs[0])

    boot_medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_medians[i] = np.median(sample)

    alpha_tail = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_medians, alpha_tail))
    hi = float(np.quantile(boot_medians, 1.0 - alpha_tail))
    return lo, hi


def effect_size_label(r: float) -> str:
    """Dien giai do lon he so tuong quan rank-biserial |r| theo nguong
    thuong dung (tham khao, khong phai chuan tuyet doi):
      |r| < 0.10        -> negligible
      0.10 <= |r| < 0.30 -> small
      0.30 <= |r| < 0.50 -> medium
      |r| >= 0.50        -> large
    """
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "n/a"
    ar = abs(r)
    if ar < 0.10:
        return "negligible"
    if ar < 0.30:
        return "small"
    if ar < 0.50:
        return "medium"
    return "large"


def benjamini_hochberg_fdr(p_values: pd.Series) -> pd.Series:
    """
    Hieu chinh p-value theo thu tuc Benjamini-Hochberg (FDR) de kiem soat
    ty le phat hien sai (false discovery rate) khi thuc hien NHIEU phep
    kiem dinh cung luc (o day: nhieu model x nhieu metric x nhieu cap pp).
    Neu khong hieu chinh, p_value < 0.05 don le rat de la duong tinh gia.

    Nhan vao 1 pd.Series p-value (co the co NaN), tra ve pd.Series cung
    index voi p-value da hieu chinh (giu nguyen NaN o vi tri NaN).
    """
    valid = p_values.dropna()
    m = len(valid)
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    if m == 0:
        return result

    sorted_idx = valid.sort_values().index
    sorted_p = valid.loc[sorted_idx].to_numpy()
    ranks = np.arange(1, m + 1)

    adjusted = sorted_p * m / ranks
    # Dam bao tinh don dieu (khong giam) khi di tu p lon nhat ve nho nhat
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result.loc[sorted_idx] = adjusted
    return result


def compare_model_metric(
    model: str,
    pp_a: str,
    a_df: pd.DataFrame,
    pp_b: str,
    b_df: pd.DataFrame,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """So sanh pp_A vs pp_B cho MOT model va MOT metric. Tra ve 1 dict = 1
    hang trong bang ket qua."""

    a_ids = set(a_df.index)
    b_ids = set(b_df.index)
    common_ids = sorted(a_ids & b_ids)
    union_ids = a_ids | b_ids

    n_common = len(common_ids)
    n_union = len(union_ids)
    pct = (n_common / n_union * 100.0) if n_union > 0 else np.nan
    common_images_str = (
        f"{n_common}/{n_union} ({pct:.2f}%)" if n_union > 0 else "0/0 (n/a)"
    )

    direction = METRIC_DIRECTION.get(metric, "higher_better")

    row = {
        "model": model,
        "pp_A": pp_a,
        "pp_B": pp_b,
        "metric": metric,
        "common_images": common_images_str,
        "p_value": np.nan,
        # cac cot bo sung
        "metric_direction": direction,
        "n_common_images": n_common,
        "n_union_images": n_union,
        "pct_common": round(pct, 2) if not np.isnan(pct) else np.nan,
        "A_median": np.nan,
        "B_median": np.nan,
        "A_std": np.nan,
        "B_std": np.nan,
        "A_iqr": np.nan,
        "B_iqr": np.nan,
        "n_pairs_used": 0,
        "wilcoxon_statistic": np.nan,
        "rank_biserial_r": np.nan,
        "effect_size_label": "n/a",
        "median_diff": np.nan,
        "median_diff_ci_low": np.nan,
        "median_diff_ci_high": np.nan,
        "significant_p<0.05": False,
        "A_better": np.nan,
    }

    has_metric = metric in a_df.columns and metric in b_df.columns
    if not has_metric or n_common == 0:
        return row

    x = a_df.loc[common_ids, metric].to_numpy(dtype=float)
    y = b_df.loc[common_ids, metric].to_numpy(dtype=float)

    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]
    if len(x) == 0:
        return row

    row["A_median"] = float(np.median(x))
    row["B_median"] = float(np.median(y))
    # Do phan tan: std (ddof=1, mau) va IQR (Q75-Q25) - IQR di kem median
    # nen ben voi outlier hon std, phu hop du lieu khong nhat thiet chuan.
    row["A_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    row["B_std"] = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    row["A_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    row["B_iqr"] = float(np.percentile(y, 75) - np.percentile(y, 25))

    diffs = x - y
    n_nonzero = int(np.sum(diffs != 0))

    try:
        if n_nonzero == 0:
            statistic, p_value = np.nan, np.nan
        else:
            statistic, p_value = wilcoxon(
                x, y, zero_method="wilcox", alternative="two-sided"
            )
    except ValueError as exc:  # noqa: BLE001
        print(f"  [canh bao] Wilcoxon loi: model={model} {pp_a} vs {pp_b} / "
              f"{metric}: {exc}")
        statistic, p_value = np.nan, np.nan

    _, _, r = rank_biserial_from_wilcoxon(diffs)
    median_diff = float(np.median(diffs)) if len(diffs) else np.nan
    significant = bool(not np.isnan(p_value) and p_value < ALPHA)

    ci_low, ci_high = bootstrap_median_diff_ci(diffs, n_boot=n_boot, rng=rng)

    if not significant or np.isnan(median_diff):
        a_better = np.nan
    elif direction == "higher_better":
        a_better = bool(median_diff > 0)
    else:
        a_better = bool(median_diff < 0)

    row.update(
        {
            "p_value": p_value,
            "n_pairs_used": n_nonzero,
            "wilcoxon_statistic": statistic,
            "rank_biserial_r": r,
            "effect_size_label": effect_size_label(r),
            "median_diff": median_diff,
            "median_diff_ci_low": ci_low,
            "median_diff_ci_high": ci_high,
            "significant_p<0.05": significant,
            "A_better": a_better,
        }
    )
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Thu muc chua cac thu muc phuong phap (mac dinh: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--proposed",
        default=DEFAULT_PROPOSED_METHOD,
        help=f"Ten thu muc phuong phap de xuat / pp A (mac dinh: {DEFAULT_PROPOSED_METHOD})",
    )
    parser.add_argument(
        "--other",
        nargs="*",
        default=None,
        help="Danh sach ten thu muc pp B can so sanh (mac dinh: TAT CA cac "
             "phuong phap con lai tim thay trong --base-dir)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Duong dan file CSV ket qua (mac dinh: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP_DEFAULT,
        help=f"So lan resample bootstrap de tinh khoang tin cay 95%% cho "
             f"median_diff (mac dinh: {N_BOOTSTRAP_DEFAULT}). Dat 0 de tat "
             f"bootstrap (chay nhanh hon nhung khong co CI).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help=f"Seed ngau nhien cho bootstrap, de ket qua tai lap duoc "
             f"(mac dinh: {BOOTSTRAP_SEED})",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_path = Path(args.output)

    print(f"Doc du lieu tu: {base_dir.resolve()}")
    methods = discover_methods(base_dir)

    if args.proposed not in methods:
        print(
            f"[loi] Khong tim thay thu muc phuong phap de xuat '{args.proposed}' "
            f"trong {base_dir}. Cac thu muc tim thay: {sorted(methods.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.other:
        missing = [m for m in args.other if m not in methods]
        if missing:
            print(f"[loi] Khong tim thay thu muc pp B: {missing}. "
                  f"Cac thu muc tim thay: {sorted(methods.keys())}", file=sys.stderr)
            sys.exit(1)
        other_methods = list(args.other)
    else:
        other_methods = [m for m in methods if m != args.proposed]

    print(f"Cac phuong phap tim thay: {sorted(methods.keys())}")
    print(f"pp A (de xuat): {args.proposed}")
    print(f"pp B (doi sanh): {other_methods}")

    # Lap chi muc model -> file cho tung phuong phap can dung (chi doc 1 lan)
    needed_methods = set(other_methods) | {args.proposed}
    method_index: dict[str, dict[str, list[Path]]] = {}
    for name in sorted(needed_methods):
        print(f"Dang lap chi muc model cho phuong phap '{name}' ...")
        method_index[name] = index_method_files(methods[name])

    # Cache du lieu da doc: (method, model) -> DataFrame
    data_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get_df(method: str, model: str) -> pd.DataFrame:
        key = (method, model)
        if key not in data_cache:
            files = method_index[method].get(model, [])
            data_cache[key] = load_csv_files(files, label=f"{method}/{model}")
        return data_cache[key]

    all_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    n_boot = args.n_bootstrap
    if n_boot > 0:
        print(f"Bootstrap CI: {n_boot} lan resample, seed={args.seed}")
    else:
        print("Bootstrap CI: TAT (n-bootstrap=0)")

    for other_name in other_methods:
        for model in MODEL_NAMES:
            a_df = get_df(args.proposed, model)
            b_df = get_df(other_name, model)
            for metric in METRIC_COLUMNS:
                all_rows.append(
                    compare_model_metric(
                        model, args.proposed, a_df, other_name, b_df, metric,
                        n_boot=n_boot, rng=rng,
                    )
                )

    if not all_rows:
        print("[canh bao] Khong co ket qua nao de xuat ra.", file=sys.stderr)
        return

    result_df = pd.DataFrame(all_rows)

    # Hieu chinh multiple comparisons (Benjamini-Hochberg FDR) tren TOAN BO
    # cac p-value trong bang - vi bang co rat nhieu phep kiem dinh (nhieu
    # model x nhieu metric x nhieu cap pp) chay cung luc, p_value tho (chua
    # hieu chinh) de bi duong tinh gia (false positive) hon.
    result_df["p_value_fdr"] = benjamini_hochberg_fdr(result_df["p_value"])
    result_df["significant_fdr_q<0.05"] = (
        result_df["p_value_fdr"].notna() & (result_df["p_value_fdr"] < ALPHA)
    )

    # Dam bao 6 cot chinh nam dau tien, dung thu tu yeu cau
    core_cols = ["model", "pp_A", "pp_B", "metric", "common_images", "p_value"]
    extra_cols = [c for c in result_df.columns if c not in core_cols]
    result_df = result_df[core_cols + extra_cols]

    result_df = result_df.sort_values(["pp_B", "model", "metric"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"\nDa xuat bang ket qua ra: {output_path.resolve()}")
    print(f"Tong so hang: {len(result_df)}")


if __name__ == "__main__":
    main()