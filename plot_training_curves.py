"""
plot_training_curves.py
=======================
Model-agnostic training curve plotter for the VS-benchmark project.
Works with any model that logs per-iteration losses in the standard
junyanz format:

    (epoch: N, iters: M, time: T, data: D) KEY1: V1 KEY2: V2 ...

Also parses:
    End of epoch N / total   Time Taken: S sec
    learning rate = V

Supported models (tested or expected):
    CUT         -- G_GAN, D_real, D_fake, G, NCE, NCE_Y
    pix2pix     -- G_GAN, G_L1, D_real, D_fake
    CycleGAN    -- D_A, G_A, cycle_A, idt_A, D_B, G_B, cycle_B, idt_B
    PSPStain    -- GAN-based, keys auto-detected
    ASP         -- GAN-based, keys auto-detected
    SwinIR / restoration models -- typically a single loss key, no GAN terms

Usage
-----
    python plot_training_curves.py \\
        --logs  path/to/run1.out  path/to/run2.out \\
        --labels  BCI  MIST-HER2 \\
        --name  CUT \\
        --out-dir  /path/to/eval-repo/results/CUT

Outputs (both written to --out-dir):
    <name>_training_curves.png   -- one figure, auto-grouped panels
    <name>_training_summary.csv  -- per-run final-epoch mean of every metric
                                    + total training time + time per epoch

Options
-------
    --logs      One or more .out (SLURM) or loss_log.txt files.
    --labels    Display names matching --logs order.
                Defaults to the file stems if omitted.
    --name      Base name used for output files. Default: "training".
    --out-dir   Directory for output files. Created if absent. Default: ".".
    --smooth    Moving-average window for loss curves. Default: 5.
    --dpi       Output image DPI. Default: 150.
    --last-n    Number of final epochs used to compute summary averages.
                Default: 10.  Use 0 to average over all epochs.

Dependencies
------------
    matplotlib  numpy  (both present in venv_cut and vs-benchmark envs)

No extra installs needed.
"""

import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches the entire "(epoch: N, iters: M, ...) KEY: V KEY: V ..." line.
ITER_HEADER_RE = re.compile(
    r"\(epoch:\s*(\d+),\s*iters:\s*\d+.*?\)"
)
# Matches every "KEY: VALUE" pair (float value) anywhere on a line.
KV_RE = re.compile(r"(\w+):\s*([\d.]+)")

EPOCH_RE = re.compile(
    r"End of epoch\s+(\d+)\s*/\s*\d+.*?Time Taken:\s*(\d+)"
)
LR_RE = re.compile(r"learning rate\s*=\s*([\d.e+\-]+)")

# Keys that appear inside the parenthesised header -- skip them.
HEADER_KEYS = {"epoch", "iters", "time", "data"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_log(path: str) -> dict:
    """
    Parse a SLURM .out file or loss_log.txt.

    Returns a dict:
        epochs          : int array of epoch numbers with iteration data
        <key>           : float array, per-epoch mean of each loss key found
        time_epochs     : int array
        time_sec        : float array
        lr_epochs       : int array
        lr              : float array
    """
    iter_buf = defaultdict(lambda: defaultdict(list))  # epoch -> key -> [values]
    epoch_time = {}
    epoch_lr = {}
    last_epoch_end = None

    with open(path, errors="replace") as fh:
        for line in fh:
            # -- iteration line --
            hdr = ITER_HEADER_RE.search(line)
            if hdr:
                epoch = int(hdr.group(1))
                # Parse every key:value pair on this line
                for key, val in KV_RE.findall(line):
                    if key.lower() not in HEADER_KEYS:
                        iter_buf[epoch][key].append(float(val))
                continue

            # -- epoch-end line --
            m = EPOCH_RE.search(line)
            if m:
                last_epoch_end = int(m.group(1))
                epoch_time[last_epoch_end] = int(m.group(2))
                continue

            # -- learning-rate line (appears right after epoch-end) --
            m = LR_RE.search(line)
            if m and last_epoch_end is not None:
                # Only record once per epoch (the line after "End of epoch N")
                if last_epoch_end not in epoch_lr:
                    epoch_lr[last_epoch_end] = float(m.group(1))

    epochs = sorted(iter_buf.keys())
    if not epochs:
        raise ValueError(f"No iteration lines found in {path}. "
                         "Check that the file is a valid training log.")

    # Collect the union of all loss keys across all epochs
    all_keys = set()
    for ep_data in iter_buf.values():
        all_keys.update(ep_data.keys())
    all_keys = sorted(all_keys)

    out = {"epochs": np.array(epochs, dtype=int)}
    for key in all_keys:
        out[key] = np.array(
            [np.mean(iter_buf[ep][key]) if iter_buf[ep][key] else float("nan")
             for ep in epochs],
            dtype=float,
        )

    ep_t = sorted(epoch_time.keys())
    out["time_epochs"] = np.array(ep_t, dtype=int)
    out["time_sec"] = np.array([epoch_time[e] for e in ep_t], dtype=float)

    ep_l = sorted(epoch_lr.keys())
    out["lr_epochs"] = np.array(ep_l, dtype=int)
    out["lr"] = np.array([epoch_lr[e] for e in ep_l], dtype=float)

    return out


# ---------------------------------------------------------------------------
# Key grouping heuristic
# ---------------------------------------------------------------------------

def group_loss_keys(all_keys_per_run: list[set]) -> list[tuple[str, list[str]]]:
    """
    Given the union of loss key sets across all runs, return a list of
    (group_title, [key, ...]) tuples for plotting.

    Grouping rules (applied in priority order):
        D-prefix or *_real/*_fake -> "Discriminator losses"
        G-prefix                  -> "Generator losses"
        NCE / nce / contrastive   -> "Contrastive losses"
        cycle / idt               -> "Cycle-consistency losses"
        L1 / l1 / recon / pixel   -> "Reconstruction losses"
        Everything else           -> "Other losses"
    """
    union: set[str] = set()
    for s in all_keys_per_run:
        union.update(s)
    union = sorted(union)

    groups: dict[str, list[str]] = {
        "Discriminator losses": [],
        "Generator losses": [],
        "Contrastive losses": [],
        "Cycle-consistency losses": [],
        "Reconstruction losses": [],
        "Other losses": [],
    }

    for key in union:
        kl = key.lower()
        if (kl.startswith("d_") or kl.startswith("d")
                and ("real" in kl or "fake" in kl)):
            groups["Discriminator losses"].append(key)
        elif kl.startswith("g_") or kl == "g":
            groups["Generator losses"].append(key)
        elif "nce" in kl or "contrastive" in kl:
            groups["Contrastive losses"].append(key)
        elif "cycle" in kl or "idt" in kl:
            groups["Cycle-consistency losses"].append(key)
        elif any(x in kl for x in ("l1", "recon", "pixel", "loss")):
            groups["Reconstruction losses"].append(key)
        else:
            groups["Other losses"].append(key)

    # Drop empty groups; preserve display order
    ordered = [
        (title, keys)
        for title, keys in groups.items()
        if keys
    ]
    return ordered


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#9333ea",
    "#ea580c", "#0891b2", "#65a30d", "#db2777",
]
LINE_STYLES = ["-", "--", ":", "-."]


def smooth_curve(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_offsets, smoothed_y) using a centred moving average."""
    if window < 2 or len(y) < window:
        return np.arange(len(y)), y
    kernel = np.ones(window) / window
    y_sm = np.convolve(y, kernel, mode="valid")
    offset = window // 2
    return np.arange(offset, offset + len(y_sm)), y_sm


def make_figure(
    datasets: list[dict],
    labels: list[str],
    loss_groups: list[tuple[str, list[str]]],
    smooth: int,
) -> plt.Figure:

    has_lr = any(len(d.get("lr", [])) > 0 for d in datasets)
    has_time = any(len(d.get("time_sec", [])) > 0 for d in datasets)

    extra_panels = int(has_lr) + int(has_time)
    n_panels = len(loss_groups) + extra_panels
    height = max(3.0, 3.5 * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=(11, height), squeeze=False)
    axes = [ax for row in axes for ax in row]  # flatten

    colors = COLORS[: len(datasets)]

    def _style_ax(ax, title):
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # -- Loss group panels --
    for ax, (group_title, keys) in zip(axes, loss_groups):
        _style_ax(ax, group_title)
        ax.set_ylabel("Loss", fontsize=9)

        is_discriminator = "Discriminator" in group_title

        for ki, key in enumerate(keys):
            ls = LINE_STYLES[ki % len(LINE_STYLES)]
            for run_data, label, color in zip(datasets, labels, colors):
                if key not in run_data:
                    continue
                x_full = run_data["epochs"]
                y_full = run_data[key]
                ax.plot(x_full, y_full, color=color, alpha=0.12,
                        linewidth=0.7, linestyle=ls)
                x_idx, y_sm = smooth_curve(y_full, smooth)
                x_sm = x_full[x_idx] if len(x_idx) == len(x_full) else x_full[: len(y_sm)]
                # align x_sm properly
                if len(x_idx) <= len(x_full):
                    x_sm = x_full[x_idx[0]: x_idx[0] + len(y_sm)]
                ax.plot(x_sm, y_sm, color=color, linewidth=1.8,
                        linestyle=ls, label=f"{label}  {key}")

        if is_discriminator:
            ax.axhline(0.25, color="gray", linestyle=":",
                       linewidth=1.0, label="ideal D = 0.25")

        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    offset = len(loss_groups)

    # -- Learning rate panel --
    if has_lr:
        ax = axes[offset]
        _style_ax(ax, "Learning rate")
        ax.set_ylabel("LR", fontsize=9)
        ax.yaxis.set_major_formatter(
            ticker.ScalarFormatter(useMathText=True)
        )
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        for run_data, label, color in zip(datasets, labels, colors):
            if len(run_data.get("lr", [])) == 0:
                continue
            ax.plot(run_data["lr_epochs"], run_data["lr"],
                    color=color, linewidth=1.8,
                    marker=".", markersize=3, label=label)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
        offset += 1

    # -- Wall-clock time panel --
    if has_time:
        ax = axes[offset]
        _style_ax(ax, "Epoch wall-clock time")
        ax.set_ylabel("sec", fontsize=9)
        for run_data, label, color in zip(datasets, labels, colors):
            if len(run_data.get("time_sec", [])) == 0:
                continue
            ax.plot(run_data["time_epochs"], run_data["time_sec"],
                    color=color, linewidth=1.5,
                    marker=".", markersize=3, label=label)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.7)

    # Collect all model names for the title
    label_str = "  |  ".join(labels)
    fig.suptitle(f"Training curves  |  {label_str}", fontsize=12, y=1.002)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary(
    datasets: list[dict],
    labels: list[str],
    last_n: int,
) -> list[dict]:
    """
    Return a list of row dicts for the summary CSV.
    One row per (label, dataset) pair.
    Columns: label, epochs_completed, total_train_time_h,
             mean_time_per_epoch_sec, then per-metric final averages.
    """
    # Collect all loss keys across all runs
    all_loss_keys: set[str] = set()
    for d in datasets:
        for k in d:
            if k not in ("epochs", "time_epochs", "time_sec",
                         "lr_epochs", "lr"):
                all_loss_keys.add(k)
    all_loss_keys = sorted(all_loss_keys)

    rows = []
    for run_data, label in zip(datasets, labels):
        epochs = run_data["epochs"]
        n_completed = int(epochs[-1]) if len(epochs) > 0 else 0

        # Training time
        total_sec = float(np.sum(run_data.get("time_sec", [])))
        mean_sec = float(np.mean(run_data["time_sec"])) \
            if len(run_data.get("time_sec", [])) > 0 else float("nan")

        row = {
            "label": label,
            "epochs_completed": n_completed,
            "total_train_time_h": round(total_sec / 3600, 3),
            "mean_time_per_epoch_sec": round(mean_sec, 1),
        }

        # Per-metric final averages
        window = last_n if last_n > 0 else len(epochs)
        for key in all_loss_keys:
            if key not in run_data:
                row[f"final_{key}"] = ""
                continue
            vals = run_data[key]
            tail = vals[-window:] if window <= len(vals) else vals
            valid = tail[~np.isnan(tail)]
            row[f"final_{key}"] = (
                round(float(np.mean(valid)), 4) if len(valid) > 0 else ""
            )

        rows.append(row)
    return rows


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model-agnostic training curve plotter for the VS benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--logs", nargs="+", required=True,
        metavar="FILE",
        help="SLURM .out or loss_log.txt files, one per run.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        metavar="LABEL",
        help="Display labels matching --logs order. "
             "Defaults to file stems.",
    )
    parser.add_argument(
        "--name", default="training",
        help="Base name for output files (default: training).",
    )
    parser.add_argument(
        "--out-dir", default=".", metavar="DIR",
        help="Output directory. Created if absent (default: current dir).",
    )
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="Moving-average window for loss curves (default: 5).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI (default: 150).",
    )
    parser.add_argument(
        "--last-n", type=int, default=10,
        metavar="N",
        help="Final N epochs used for summary averages (default: 10). "
             "Use 0 for all epochs.",
    )
    args = parser.parse_args()

    # Resolve labels
    labels = args.labels if args.labels else [Path(p).stem for p in args.logs]
    if len(labels) != len(args.logs):
        parser.error("--labels count must match --logs count.")

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse all logs
    datasets = []
    all_key_sets = []
    for path, label in zip(args.logs, labels):
        print(f"Parsing  {path}  ({label}) ...", end=" ", flush=True)
        try:
            d = parse_log(path)
        except ValueError as exc:
            print(f"\nERROR: {exc}")
            raise SystemExit(1)
        n_ep = len(d["epochs"])
        loss_keys = [k for k in d if k not in
                     ("epochs", "time_epochs", "time_sec", "lr_epochs", "lr")]
        print(f"{n_ep} epochs,  keys: {', '.join(sorted(loss_keys))}")
        datasets.append(d)
        all_key_sets.append(set(loss_keys))

    # Group keys for plot panels
    loss_groups = group_loss_keys(all_key_sets)
    if not loss_groups:
        print("WARNING: No loss keys detected. "
              "The log may not contain standard iteration lines.")

    # Plot
    fig = make_figure(datasets, labels, loss_groups, args.smooth)
    plot_path = out_dir / f"{args.name}_training_curves.png"
    fig.savefig(plot_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved   : {plot_path}")

    # Summary CSV
    summary_rows = build_summary(datasets, labels, args.last_n)
    csv_path = out_dir / f"{args.name}_training_summary.csv"
    write_summary_csv(summary_rows, csv_path)
    print(f"Summary CSV  : {csv_path}")

    # Print summary to stdout as well
    print(f"\n{'Label':<20} {'Epochs':>7} {'Total (h)':>10} "
          f"{'Sec/epoch':>10}")
    print("-" * 52)
    for row in summary_rows:
        print(f"{row['label']:<20} {row['epochs_completed']:>7} "
              f"{row['total_train_time_h']:>10.2f} "
              f"{row['mean_time_per_epoch_sec']:>10.1f}")


if __name__ == "__main__":
    main()