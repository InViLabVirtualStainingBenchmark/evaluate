"""
Benchmark evaluation script for virtual staining histopathology image-to-image models.

Computes PSNR, SSIM, MS-SSIM, LPIPS, MAE, and FID between a folder of predicted images
and a folder of ground truth images. Optionally runs Cellpose segmentation on both folders
to compute downstream cell-detection precision, recall, and F1 (enable with --cellpose).
Logs library versions and a GT folder checksum for full reproducibility. Appends results
to a CSV file if --output is provided.

Resolution policy: when predicted and ground-truth image sizes differ, the GT image is
downsampled to match the predicted resolution (LANCZOS). This keeps evaluation at the
model's native output resolution and avoids upscaling artifacts.
"""

from __future__ import annotations

# stdlib
import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import logging
import os
import pathlib
import random
import sys

# third-party
try:
    import numpy as np
except ImportError:
    sys.stderr.write("Missing library: numpy. Install with: pip install numpy\n")
    sys.exit(1)

try:
    import torch
except ImportError:
    sys.stderr.write("Missing library: torch. Install with: pip install torch\n")
    sys.exit(1)

try:
    import torchmetrics
    import torchmetrics.functional
except ImportError:
    sys.stderr.write("Missing library: torchmetrics. Install with: pip install torchmetrics\n")
    sys.exit(1)

try:
    import lpips
except ImportError:
    sys.stderr.write("Missing library: lpips. Install with: pip install lpips\n")
    sys.exit(1)

try:
    from torch_fidelity import calculate_metrics as fidelity_calculate_metrics
except ImportError:
    sys.stderr.write(
        "Missing library: torch-fidelity. Install with: pip install torch-fidelity\n"
    )
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    sys.stderr.write("Missing library: Pillow. Install with: pip install Pillow\n")
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predicted images against ground truth using PSNR, SSIM, "
            "MS-SSIM, LPIPS, MAE, and FID."
        )
    )
    parser.add_argument(
        "--pred", type=str, required=True,
        help="Folder of predicted / generated images.",
    )
    parser.add_argument(
        "--gt", type=str, required=True,
        help="Folder of ground truth images.",
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name label for this model run, used in output.",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="Name label for the dataset used.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to CSV file to append results to. File is created if it does not exist.",
    )
    parser.add_argument(
        "--match_by", type=str, default="sort", choices=["sort", "stem"],
        help=(
            "How to pair pred and gt images. "
            "'sort': match by alphabetical position (folders must have equal counts). "
            "'stem': match by filename stem after stripping _fake_B/_real_B/_fake/_real suffixes."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use for metric computation: 'cuda', 'cpu', or 'auto'.",
    )
    parser.add_argument(
        "--runtime", type=float, default=None,
        help="Inference runtime in seconds (pass-through value from job script).",
    )
    parser.add_argument(
        "--gpu_mem", type=float, default=None,
        help="Peak GPU memory in MB during inference (pass-through value from job script).",
    )
    parser.add_argument(
        "--split_name", type=str, default="test",
        help="Name of the dataset split being evaluated (e.g. 'test', 'val').",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pred_suffix", type=str, default=None,
        help=(
            "If set, only pred images whose stem ends with this suffix are considered. "
            "Useful when the pred folder contains mixed image types (e.g. '--pred_suffix _fake_B' "
            "for pytorch-CycleGAN-and-pix2pix results folders)."
        ),
    )
    parser.add_argument(
        "--cellpose", action="store_true", default=False,
        help="Enable Cellpose cell segmentation evaluation (opt-in; requires cellpose installed).",
    )
    parser.add_argument(
        "--cellpose_model", type=str, default="cyto2",
        help=(
            "Cellpose model type to use when --cellpose is set. "
            "Common options: 'cyto2' (H&E / cytoplasm), 'nuclei' (DAPI / nuclear), 'cyto'. "
            "Default: cyto2."
        ),
    )
    parser.add_argument(
        "--cellpose_n", type=int, default=None,
        help=(
            "Number of image pairs to run Cellpose on. "
            "If not set, all pairs are evaluated. "
            "The subset is drawn with the global --seed, so results are fully reproducible."
        ),
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    """Set random seeds for Python, numpy, and torch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_library_versions(cellpose_enabled: bool = False) -> dict[str, str]:
    """Collect runtime version strings for all metric libraries via importlib.metadata."""
    lib_names = [
        "torch", "torchmetrics", "lpips", "torch-fidelity",
        "numpy", "Pillow",
    ]
    if cellpose_enabled:
        lib_names.append("cellpose")
    versions: dict[str, str] = {}
    for lib in lib_names:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "unknown"
    return versions


def compute_gt_checksum(gt_folder: str) -> str:
    """Compute MD5 of GT folder by hashing all image file contents in sorted filename order."""
    gt_path = pathlib.Path(gt_folder)
    files = sorted(
        f for f in gt_path.iterdir()
        if f.suffix.lower() in _IMAGE_EXTENSIONS and not f.name.startswith(".")
    )
    md5 = hashlib.md5()
    for f in files:
        md5.update(f.read_bytes())
    return md5.hexdigest()


def collect_image_pairs(
    pred_folder: str, gt_folder: str, match_by: str, pred_suffix: str | None = None
) -> list[tuple[str, str]]:
    """Return (pred_path, gt_path) tuples using 'sort' or 'stem' pairing strategy."""
    def list_images(folder: str, suffix_filter: str | None = None) -> list[pathlib.Path]:
        p = pathlib.Path(folder)
        files = sorted(
            f for f in p.iterdir()
            if f.suffix.lower() in _IMAGE_EXTENSIONS and not f.name.startswith(".")
        )
        if suffix_filter is not None:
            files = [f for f in files if f.stem.endswith(suffix_filter)]
        return files

    pred_files = list_images(pred_folder, pred_suffix)
    gt_files = list_images(gt_folder)

    if match_by == "sort":
        if len(pred_files) != len(gt_files):
            sys.stderr.write(
                f"ERROR: --match_by sort requires equal image counts. "
                f"pred has {len(pred_files)}, gt has {len(gt_files)}.\n"
            )
            sys.exit(1)
        pairs = [(str(p), str(g)) for p, g in zip(pred_files, gt_files)]

    else:  # stem
        stem_suffixes = ["_fake_B", "_real_B", "_fake", "_real"]

        def strip_stem(stem: str) -> str:
            for suffix in stem_suffixes:
                if stem.endswith(suffix):
                    return stem[: -len(suffix)]
            return stem

        pred_map: dict[str, str] = {strip_stem(pf.stem): str(pf) for pf in pred_files}

        pairs = []
        missing: list[str] = []
        for gf in gt_files:
            base = gf.stem
            if base not in pred_map:
                missing.append(base)
            else:
                pairs.append((pred_map[base], str(gf)))

        if missing:
            preview = missing[:5]
            ellipsis = "..." if len(missing) > 5 else ""
            sys.stderr.write(
                f"ERROR: {len(missing)} GT stem(s) have no matching predicted image: "
                f"{preview}{ellipsis}\n"
            )
            sys.exit(1)

    _log.info("Found %d image pairs.", len(pairs))
    return pairs


def load_image_pair(
    pred_path: str, gt_path: str, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a pred/gt pair as float32 tensors of shape (1, 3, H, W) in range [0, 1].

    When sizes differ, the GT image is downsampled to match the predicted resolution
    (LANCZOS). This keeps evaluation at the model's native output resolution and avoids
    bilinear upscaling artifacts being attributed to the model.
    """
    def open_image(path: str) -> tuple[Image.Image, float]:
        img = Image.open(path)
        if img.mode in ("I", "I;16"):
            _log.warning("16-bit image detected: %s. Normalizing by 65535.0.", path)
            divisor = 65535.0
        else:
            divisor = 255.0
        return img.convert("RGB"), divisor

    pred_img, pred_div = open_image(pred_path)
    gt_img, gt_div = open_image(gt_path)

    if pred_img.size != gt_img.size:
        # Downsample GT to pred resolution. Never upscale pred, as that would
        # introduce blur artifacts that would unfairly penalize the model.
        _log.warning(
            "Size mismatch: pred '%s' is %s, gt '%s' is %s. "
            "Downsampling gt to pred resolution (LANCZOS).",
            pred_path, pred_img.size, gt_path, gt_img.size,
        )
        gt_img = gt_img.resize(pred_img.size, Image.LANCZOS)

    def to_tensor(img: Image.Image, divisor: float) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / divisor  # (H, W, 3)
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
        return t.to(device)

    return to_tensor(pred_img, pred_div), to_tensor(gt_img, gt_div)


def compute_per_image_metrics(
    pairs: list[tuple[str, str]], device: str
) -> dict[str, list[float]]:
    """Compute PSNR, SSIM, MS-SSIM, LPIPS (AlexNet and VGG), and MAE for every image pair."""
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)
    loss_fn_alex.eval()
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
    loss_fn_vgg.eval()

    results: dict[str, list[float]] = {
        "psnr": [], "ssim": [], "ms_ssim": [], "lpips_alex": [], "lpips_vgg": [], "mae": [],
    }

    with torch.no_grad():
        for pred_path, gt_path in pairs:
            pred_t, gt_t = load_image_pair(pred_path, gt_path, device)

            psnr_val = torchmetrics.functional.peak_signal_noise_ratio(
                pred_t, gt_t, data_range=1.0
            ).item()
            results["psnr"].append(psnr_val)

            ssim_val = torchmetrics.functional.structural_similarity_index_measure(
                pred_t, gt_t, data_range=1.0
            ).item()
            results["ssim"].append(ssim_val)

            ms_ssim_val = (
                torchmetrics.functional.multiscale_structural_similarity_index_measure(
                    pred_t, gt_t, data_range=1.0
                )
            ).item()
            results["ms_ssim"].append(ms_ssim_val)

            pred_lpips = pred_t * 2.0 - 1.0
            gt_lpips = gt_t * 2.0 - 1.0
            results["lpips_alex"].append(loss_fn_alex(pred_lpips, gt_lpips).item())
            results["lpips_vgg"].append(loss_fn_vgg(pred_lpips, gt_lpips).item())

            pred_np = pred_t.cpu().numpy()
            gt_np = gt_t.cpu().numpy()
            mae_val = float(np.mean(np.abs(pred_np - gt_np)))
            results["mae"].append(mae_val)

    return results


def compute_fid(pred_folder: str, gt_folder: str) -> float:
    """Compute Frechet Inception Distance between predicted and ground truth image folders."""
    metrics = fidelity_calculate_metrics(
        input1=pred_folder,
        input2=gt_folder,
        cuda=torch.cuda.is_available(),
        fid=True,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def _compare_instance_masks(
    pred_mask: np.ndarray, gt_mask: np.ndarray, iou_threshold: float = 0.5
) -> tuple[float, float, float]:
    """Return (precision, recall, F1) by greedy IoU matching of Cellpose instance masks."""
    gt_ids = np.unique(gt_mask)
    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred_mask)
    pred_ids = pred_ids[pred_ids != 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_ids) == 0:
        return 0.0, 1.0, 0.0
    if len(pred_ids) == 0:
        return 1.0, 0.0, 0.0

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, g in enumerate(gt_ids):
        gt_cell = gt_mask == g
        for j, p in enumerate(pred_ids):
            pred_cell = pred_mask == p
            intersection = int(np.logical_and(gt_cell, pred_cell).sum())
            if intersection == 0:
                continue
            union = int(np.logical_or(gt_cell, pred_cell).sum())
            iou_matrix[i, j] = intersection / union

    candidate_indices = np.argwhere(iou_matrix >= iou_threshold)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    tp = 0

    if len(candidate_indices) > 0:
        candidate_ious = iou_matrix[candidate_indices[:, 0], candidate_indices[:, 1]]
        for idx in np.argsort(-candidate_ious):
            gi, pi = candidate_indices[idx]
            if gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(int(gi))
                matched_pred.add(int(pi))
                tp += 1

    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def compute_cellpose_metrics(
    pairs: list[tuple[str, str]],
    device: str,
    model_type: str,
) -> dict[str, list[float]]:
    """Run Cellpose on the given pairs and return per-image precision, recall, F1, and cell counts."""
    from cellpose import models as cellpose_models  # noqa: PLC0415 — lazy optional import

    use_gpu = device == "cuda"
    cp_model = cellpose_models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)
    _log.info("Running Cellpose (%s) on %d pairs (device=%s).", model_type, len(pairs), device)

    results: dict[str, list[float]] = {
        "cp_precision": [], "cp_recall": [], "cp_f1": [],
        "cp_n_pred": [], "cp_n_gt": [],
    }

    for pred_path, gt_path in pairs:
        try:
            pred_img = np.array(Image.open(pred_path).convert("RGB"))
            gt_img = np.array(Image.open(gt_path).convert("RGB"))
            if pred_img.shape != gt_img.shape:
                # Downsample GT to pred resolution, consistent with load_image_pair.
                # numpy shape is (H, W, C); PIL resize takes (W, H).
                _log.warning(
                    "Cellpose size mismatch: pred %s is %s, gt %s is %s. "
                    "Downsampling gt to pred resolution (LANCZOS).",
                    pred_path, pred_img.shape[:2], gt_path, gt_img.shape[:2],
                )
                gt_img = np.array(
                    Image.fromarray(gt_img).resize(
                        (pred_img.shape[1], pred_img.shape[0]), Image.LANCZOS
                    )
                )
            pred_masks = cp_model.eval(pred_img, diameter=None, channels=[0, 0])[0]
            gt_masks = cp_model.eval(gt_img, diameter=None, channels=[0, 0])[0]
            precision, recall, f1 = _compare_instance_masks(pred_masks, gt_masks)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Cellpose skipping pair (%s, %s): %s", pred_path, gt_path, exc)
            continue

        results["cp_precision"].append(precision)
        results["cp_recall"].append(recall)
        results["cp_f1"].append(f1)
        results["cp_n_pred"].append(float(int(pred_masks.max())))
        results["cp_n_gt"].append(float(int(gt_masks.max())))

    return results


def format_terminal_output(
    model_name: str,
    dataset_name: str,
    split_name: str,
    timestamp: str,
    num_pairs: int,
    gt_checksum: str,
    per_image: dict[str, list[float]],
    fid: float,
    runtime: float | None,
    gpu_mem: float | None,
    versions: dict[str, str],
    output_path: str | None,
    cellpose_results: dict[str, list[float]] | None = None,
    cellpose_model: str | None = None,
    cellpose_n_sampled: int | None = None,
) -> str:
    """Build the formatted evaluation results string; does not print."""
    SEP = "=" * 60
    COL1_DASH = "-" * 14
    COL2_DASH = "-" * 8

    def mean_std(vals: list[float]) -> tuple[float, float]:
        return float(np.mean(vals)), float(np.std(vals))

    psnr_m, psnr_s = mean_std(per_image["psnr"])
    ssim_m, ssim_s = mean_std(per_image["ssim"])
    ms_ssim_m, ms_ssim_s = mean_std(per_image["ms_ssim"])
    lpips_alex_m, lpips_alex_s = mean_std(per_image["lpips_alex"])
    lpips_vgg_m, lpips_vgg_s = mean_std(per_image["lpips_vgg"])
    mae_m, mae_s = mean_std(per_image["mae"])

    def row(name: str, mean_str: str, std_str: str) -> str:
        return f"  {name:<18}{mean_str:<12}{std_str}"

    metric_rows = [
        row("PSNR",            f"{psnr_m:.2f} dB",    f"{psnr_s:.2f}"),
        row("SSIM",            f"{ssim_m:.3f}",        f"{ssim_s:.3f}"),
        row("MS-SSIM",         f"{ms_ssim_m:.3f}",     f"{ms_ssim_s:.3f}"),
        row("LPIPS (AlexNet)", f"{lpips_alex_m:.3f}",  f"{lpips_alex_s:.3f}"),
        row("LPIPS (VGG)",     f"{lpips_vgg_m:.3f}",   f"{lpips_vgg_s:.3f}"),
        row("MAE",             f"{mae_m:.3f}",         f"{mae_s:.3f}"),
        row("FID",             f"{fid:.2f}",           "--"),
        row("Runtime",         f"{runtime:.1f} s"  if runtime  is not None else "--", "--"),
        row("GPU Memory",      f"{gpu_mem:.0f} MB" if gpu_mem  is not None else "--", "--"),
    ]

    version_libs = ["torch", "torchmetrics", "lpips", "torch-fidelity", "numpy", "Pillow"]
    if cellpose_results is not None:
        version_libs.append("cellpose")
    version_rows = [
        f"  {lib:<18}{versions.get(lib, 'unknown')}" for lib in version_libs
    ]

    header_sep = f"  {COL1_DASH}    {COL2_DASH}    {COL2_DASH}"

    lines: list[str] = [
        SEP,
        f"  Evaluation: {model_name} on {dataset_name} ({split_name})",
        f"  Run at: {timestamp} UTC",
        f"  Pairs evaluated: {num_pairs}",
        f"  GT checksum (MD5): {gt_checksum}",
        SEP,
        f"  {'Metric':<18}{'Mean':<12}Std",
        header_sep,
        *metric_rows,
        SEP,
    ]

    if cellpose_results is not None and cellpose_results.get("cp_f1"):
        cp_m_std = lambda key: (  # noqa: E731
            float(np.mean(cellpose_results[key])),
            float(np.std(cellpose_results[key])),
        )
        cp_prec_m, cp_prec_s = cp_m_std("cp_precision")
        cp_rec_m, cp_rec_s = cp_m_std("cp_recall")
        cp_f1_m, cp_f1_s = cp_m_std("cp_f1")
        n_label = f"{cellpose_n_sampled} pairs sampled" if cellpose_n_sampled is not None else "all pairs"
        lines += [
            f"  Cellpose ({cellpose_model}) -- {n_label}",
            f"  {'Metric':<18}{'Mean':<12}Std",
            header_sep,
            row("CP Precision",    f"{cp_prec_m:.3f}", f"{cp_prec_s:.3f}"),
            row("CP Recall",       f"{cp_rec_m:.3f}",  f"{cp_rec_s:.3f}"),
            row("CP F1",           f"{cp_f1_m:.3f}",   f"{cp_f1_s:.3f}"),
            SEP,
        ]

    lines += [
        "  Library versions",
        f"  {COL1_DASH}",
        *version_rows,
        SEP,
    ]

    if output_path is not None:
        lines.append(f"  Saved to: {output_path}")
        lines.append(SEP)

    return "\n".join(lines)


def append_csv_row(
    output_path: str,
    model_name: str,
    dataset_name: str,
    split_name: str,
    timestamp: str,
    num_pairs: int,
    per_image: dict[str, list[float]],
    fid: float,
    runtime: float | None,
    gpu_mem: float | None,
    seed: int,
    gt_checksum: str,
    versions: dict[str, str],
    cellpose_results: dict[str, list[float]] | None = None,
    cellpose_model: str | None = None,
    cellpose_n_sampled: int | None = None,
) -> None:
    """Append one result row to the CSV file, writing the header first if needed."""
    columns = [
        "timestamp_utc", "model_name", "dataset_name", "split_name", "num_pairs",
        "psnr_mean", "psnr_std",
        "ssim_mean", "ssim_std",
        "ms_ssim_mean", "ms_ssim_std",
        "lpips_alex_mean", "lpips_alex_std",
        "lpips_vgg_mean", "lpips_vgg_std",
        "mae_mean", "mae_std",
        "fid",
        "runtime_s", "gpu_mem_mb",
        "seed",
        "gt_checksum_md5",
        "torch_version", "torchmetrics_version", "lpips_version",
        "torch_fidelity_version", "numpy_version", "pillow_version",
        "cellpose_model", "cellpose_n_pairs",
        "cellpose_precision_mean", "cellpose_precision_std",
        "cellpose_recall_mean", "cellpose_recall_std",
        "cellpose_f1_mean", "cellpose_f1_std",
        "cellpose_version",
    ]

    def fmt(val: float | None) -> str:
        return "N/A" if val is None else f"{val:.6f}"

    def mean_std(vals: list[float]) -> tuple[float, float]:
        return float(np.mean(vals)), float(np.std(vals))

    psnr_m, psnr_s = mean_std(per_image["psnr"])
    ssim_m, ssim_s = mean_std(per_image["ssim"])
    ms_ssim_m, ms_ssim_s = mean_std(per_image["ms_ssim"])
    lpips_alex_m, lpips_alex_s = mean_std(per_image["lpips_alex"])
    lpips_vgg_m, lpips_vgg_s = mean_std(per_image["lpips_vgg"])
    mae_m, mae_s = mean_std(per_image["mae"])

    row_data = {
        "timestamp_utc":          timestamp,
        "model_name":             model_name,
        "dataset_name":           dataset_name,
        "split_name":             split_name,
        "num_pairs":              str(num_pairs),
        "psnr_mean":              fmt(psnr_m),
        "psnr_std":               fmt(psnr_s),
        "ssim_mean":              fmt(ssim_m),
        "ssim_std":               fmt(ssim_s),
        "ms_ssim_mean":           fmt(ms_ssim_m),
        "ms_ssim_std":            fmt(ms_ssim_s),
        "lpips_alex_mean":        fmt(lpips_alex_m),
        "lpips_alex_std":         fmt(lpips_alex_s),
        "lpips_vgg_mean":         fmt(lpips_vgg_m),
        "lpips_vgg_std":          fmt(lpips_vgg_s),
        "mae_mean":               fmt(mae_m),
        "mae_std":                fmt(mae_s),
        "fid":                    fmt(fid),
        "runtime_s":              fmt(runtime),
        "gpu_mem_mb":             fmt(gpu_mem),
        "seed":                   str(seed),
        "gt_checksum_md5":        gt_checksum,
        "torch_version":          versions.get("torch", "N/A"),
        "torchmetrics_version":   versions.get("torchmetrics", "N/A"),
        "lpips_version":          versions.get("lpips", "N/A"),
        "torch_fidelity_version": versions.get("torch-fidelity", "N/A"),
        "numpy_version":          versions.get("numpy", "N/A"),
        "pillow_version":         versions.get("Pillow", "N/A"),
    }

    if cellpose_results is not None and cellpose_results.get("cp_f1"):
        def cp_mean_std(key: str) -> tuple[float, float]:
            vals = cellpose_results[key]
            return float(np.mean(vals)), float(np.std(vals))

        cp_prec_m, cp_prec_s = cp_mean_std("cp_precision")
        cp_rec_m, cp_rec_s = cp_mean_std("cp_recall")
        cp_f1_m, cp_f1_s = cp_mean_std("cp_f1")
        row_data.update({
            "cellpose_model":           cellpose_model or "N/A",
            "cellpose_n_pairs":         str(cellpose_n_sampled) if cellpose_n_sampled is not None else "N/A",
            "cellpose_precision_mean":  fmt(cp_prec_m),
            "cellpose_precision_std":   fmt(cp_prec_s),
            "cellpose_recall_mean":     fmt(cp_rec_m),
            "cellpose_recall_std":      fmt(cp_rec_s),
            "cellpose_f1_mean":         fmt(cp_f1_m),
            "cellpose_f1_std":          fmt(cp_f1_s),
            "cellpose_version":         versions.get("cellpose", "N/A"),
        })
    else:
        row_data.update({
            "cellpose_model":           "N/A",
            "cellpose_n_pairs":         "N/A",
            "cellpose_precision_mean":  "N/A",
            "cellpose_precision_std":   "N/A",
            "cellpose_recall_mean":     "N/A",
            "cellpose_recall_std":      "N/A",
            "cellpose_f1_mean":         "N/A",
            "cellpose_f1_std":          "N/A",
            "cellpose_version":         "N/A",
        })

    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def main() -> None:
    """Parse arguments, run evaluation pipeline, print results, and optionally save CSV."""
    args = parse_args()

    set_seeds(args.seed)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            sys.stderr.write(
                "ERROR: --device cuda was requested but CUDA is not available on this machine.\n"
            )
            sys.exit(1)
        device = "cuda"
    else:
        device = "cpu"

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    versions = get_library_versions(cellpose_enabled=args.cellpose)

    # Validate pred and gt folders
    for label, folder in [("pred", args.pred), ("gt", args.gt)]:
        p = pathlib.Path(folder)
        if not p.exists() or not p.is_dir():
            sys.stderr.write(f"ERROR: --{label} folder does not exist: {folder}\n")
            sys.exit(1)
        images = [
            f for f in p.iterdir()
            if f.suffix.lower() in _IMAGE_EXTENSIONS and not f.name.startswith(".")
        ]
        if not images:
            sys.stderr.write(
                f"ERROR: --{label} folder contains no supported images "
                f"(.png/.jpg/.jpeg/.tif/.tiff): {folder}\n"
            )
            sys.exit(1)

    gt_checksum = compute_gt_checksum(args.gt)

    pairs = collect_image_pairs(args.pred, args.gt, args.match_by, args.pred_suffix)

    if len(pairs) < 2048:
        _log.warning(
            "FID is computed on %d pairs. Statistical reliability of FID generally "
            "requires at least 2048 samples. Interpret this value with caution.",
            len(pairs),
        )

    per_image = compute_per_image_metrics(pairs, device)

    fid = compute_fid(args.pred, args.gt)

    cellpose_results: dict[str, list[float]] | None = None
    cellpose_pairs_sampled: list[tuple[str, str]] | None = None
    if args.cellpose:
        try:
            import cellpose  # noqa: F401 — verify importable before proceeding
        except ImportError:
            sys.stderr.write(
                "Missing library: cellpose. Install with: pip install cellpose\n"
            )
            sys.exit(1)
        if args.cellpose_n is not None:
            n = min(args.cellpose_n, len(pairs))
            cellpose_pairs_sampled = random.sample(pairs, n)
        else:
            cellpose_pairs_sampled = list(pairs)
        cellpose_results = compute_cellpose_metrics(
            cellpose_pairs_sampled, device, args.cellpose_model
        )

    cp_n_sampled = len(cellpose_pairs_sampled) if cellpose_pairs_sampled is not None else None
    output_str = format_terminal_output(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        timestamp=timestamp,
        num_pairs=len(pairs),
        gt_checksum=gt_checksum,
        per_image=per_image,
        fid=fid,
        runtime=args.runtime,
        gpu_mem=args.gpu_mem,
        versions=versions,
        output_path=args.output,
        cellpose_results=cellpose_results,
        cellpose_model=args.cellpose_model if args.cellpose else None,
        cellpose_n_sampled=cp_n_sampled,
    )
    print(output_str)

    if args.output is not None:
        pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        append_csv_row(
            output_path=args.output,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            split_name=args.split_name,
            timestamp=timestamp,
            num_pairs=len(pairs),
            per_image=per_image,
            fid=fid,
            runtime=args.runtime,
            gpu_mem=args.gpu_mem,
            seed=args.seed,
            gt_checksum=gt_checksum,
            versions=versions,
            cellpose_results=cellpose_results,
            cellpose_model=args.cellpose_model if args.cellpose else None,
            cellpose_n_sampled=cp_n_sampled,
        )


if __name__ == "__main__":
    main()