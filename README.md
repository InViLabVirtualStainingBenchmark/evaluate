# Virtual Staining Benchmark - Evaluation Script

A single, model-agnostic script that evaluates any image-to-image translation model by
comparing its output images against ground truth. Designed to produce a reproducible
comparison table across all models, datasets, and runs.

---

## Why this exists

Every paper reports metrics differently: different datasets, different library versions,
sometimes different definitions of the same metric. This script solves that by running
every model through the same evaluation code, the same library versions, and logging
everything needed to reproduce a result: a GT folder checksum, a UTC timestamp, all
library versions, and the random seed.

---

## Setup

Create and activate the benchmark conda environment:

```bash
conda env create -f environment.yml
conda activate vs-benchmark
```

Or install with pip (install torch separately first : see comment in requirements.txt):

```bash
pip install -r requirements.txt
```

---

## Basic usage

After a model has finished inference and written its output images to a folder, run:

```bash
python evaluate.py \
  --pred path/to/predicted/images \
  --gt path/to/ground_truth/images \
  --model_name my_model \
  --dataset_name my_dataset \
  --output results/results.csv
```

Results are printed to the terminal and appended as one row to the CSV.

---

## Arguments

| Argument           | Required | Default    | Description                                                           |
|--------------------|----------|------------|-----------------------------------------------------------------------|
| `--pred`           | yes      | :          | Folder of predicted / generated images                                |
| `--gt`             | yes      | :          | Folder of ground truth images                                         |
| `--model_name`     | yes      | :          | Label for this model, used in output                                  |
| `--dataset_name`   | yes      | :          | Label for the dataset used                                            |
| `--output`         | no       | None       | CSV file to append results to (created if missing)                    |
| `--match_by`       | no       | `sort`     | How to pair pred and gt images: `sort` or `stem`                      |
| `--pred_suffix`    | no       | None       | Only use pred images whose filename ends with this suffix             |
| `--device`         | no       | `auto`     | `cuda`, `cpu`, or `auto`                                              |
| `--runtime`        | no       | None       | Inference time in seconds (pass-through from job script)              |
| `--gpu_mem`        | no       | None       | Peak GPU memory in MB (pass-through from job script)                  |
| `--split_name`     | no       | `test`     | Dataset split name, e.g. `test` or `val`                              |
| `--seed`           | no       | `42`       | Random seed for reproducibility                                       |
| `--cellpose`       | no       | off        | Enable Cellpose cell segmentation evaluation (see below)              |
| `--cellpose_model` | no       | `cpsam`    | Cellpose model: `cpsam`, `cyto2` (H&E), `nuclei`                      |
| `--cellpose_n`     | no       | None (all) | Number of pairs to run Cellpose on; subset is seeded and reproducible |

---

## Image matching

The script needs to know which predicted image corresponds to which ground truth image.
Two strategies are available via `--match_by`:

**`sort` (default)**
Sorts both folders alphabetically and matches by position. Both folders must have the
same number of images or the script exits with an error. Use this when your model writes
one clean output per input with no extra files.

**`stem`**
Matches images by filename stem. Before matching, it strips known suffixes from predicted
filenames: `_fake_B`, `_real_B`, `_fake`, `_real`. So `0001_fake_B.png` matches `0001.png`.
Use this when filenames correspond but have different suffixes or extensions.

---

## Handling mixed output folders

Some models (e.g. pytorch-CycleGAN-and-pix2pix) write multiple image types into one
folder: `_fake_B`, `_real_A`, `_real_B`, `_rec_A`, etc. Passing such a folder directly
as `--pred` would cause incorrect matches.

Use `--pred_suffix` to filter to only the images you want:

```bash
python evaluate.py \
  --pred results/facades_pix2pix/test_latest/images \
  --gt datasets/facades/testB \
  --model_name facades_pix2pix \
  --dataset_name facades \
  --match_by stem \
  --pred_suffix _fake_B
```

Models whose results folder already contains only one image type (e.g. CUT, SwinIR,
Restormer) do not need this flag.

---

## Metrics

### Per-image metrics
Computed for every matched pair. Mean and standard deviation are reported.

| Metric          | Better | Notes                                                                                                                                     |
|-----------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------|
| PSNR            | higher | Peak signal-to-noise ratio in dB. Sensitive to pixel-level accuracy.                                                                      |
| SSIM            | higher | Structural similarity, range 0-1.                                                                                                         |
| MS-SSIM         | higher | Multi-scale SSIM, range 0-1.                                                                                                              |
| LPIPS (AlexNet) | lower  | Perceptual similarity using AlexNet. Best standalone perceptual metric.                                                                   |
| LPIPS (VGG)     | lower  | Perceptual similarity using VGG. Common in GAN training pipelines. Both are reported because the field has not converged on one standard. |
| MAE             | lower  | Mean absolute pixel error.                                                                                                                |

### Distribution-level metric
Computed once across the exact matched image pairs used for the per-image metrics.

| Metric | Better | Notes                                                                                                                                                                                                                                          |
|--------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FID    | lower  | Frechet Inception Distance. Measures how similar the overall distribution of generated images is to real images. Most meaningful for unpaired models. Requires at least 2048 images for reliable results. A warning is shown for smaller sets. |

### Pass-through metrics
Not computed by the script. Provide them from your job script if you have them.

| Metric            | Argument    |
|-------------------|-------------|
| Inference runtime | `--runtime` |
| Peak GPU memory   | `--gpu_mem` |

### Cellpose segmentation metrics (optional)

Enable with `--cellpose`. Requires `pip install cellpose`.

Cellpose is run independently on each predicted image and its ground truth counterpart to
produce instance segmentation masks (one integer label per cell, 0 = background). The masks
are then compared using greedy IoU matching at a threshold of 0.5: a predicted cell counts
as a true positive only if it overlaps its best-matching ground truth cell by at least 50%.

This is a **downstream-task metric**: instead of asking "do the pixel values match?", it
asks "does a cell segmentation algorithm behave the same way on the generated image as on
the real one?" A model can score well on PSNR/SSIM but still fool a segmenter differently
than the ground truth stain would : or vice versa.

| Metric       | Better | Notes                                                   |
|--------------|--------|---------------------------------------------------------|
| CP Precision | higher | Fraction of detected cells in pred that match a GT cell |
| CP Recall    | higher | Fraction of GT cells that are matched by a pred cell    |
| CP F1        | higher | Harmonic mean of precision and recall                   |

**Choosing a model**: pass the model name that matches your staining type:

| Staining                | `--cellpose_model` |
|-------------------------|--------------------|
| General histology       | `cpsam` (default)  |
| H&E (cytoplasm)         | `cyto2`            |
| DAPI / Hoechst (nuclei) | `nuclei`           |
| Generic cytoplasm       | `cyto`             |

**Sampling**: on large datasets Cellpose can be slow. Use `--cellpose_n N` to run on a
random subset of N pairs instead of all pairs. The subset is drawn using Python's `random`
module after the global `--seed` has been set, so the same seed always produces the same
subset. The exact count is written to the CSV under `cellpose_n_pairs`.

```bash
python evaluate.py \
  --pred results/my_model/images \
  --gt datasets/test \
  --model_name my_model \
  --dataset_name my_dataset \
  --cellpose \
  --cellpose_model nuclei \
  --cellpose_n 200
```

**Cluster / HPC usage**: compute nodes typically have no internet access. Cellpose
downloads model weights from HuggingFace on first use and caches them in
`~/.cellpose/models/`. Pre-download on the login node before submitting any eval job:

```bash
python -c "
from cellpose import models
models.CellposeModel(pretrained_model='cpsam')   # default
# models.CellposeModel(pretrained_model='cyto2')  # add if using --cellpose_model cyto2
# models.CellposeModel(pretrained_model='nuclei') # add if using --cellpose_model nuclei
"
```

This is already handled by `install_eval.sh` (pre-downloads `cpsam`). If you switch to
a different model type, add the corresponding line there and re-run the installation job.

---

## Terminal output

Without `--cellpose`:

```
============================================================
  Evaluation: maps_CUT on maps (test)
  Run at: 2026-04-01 10:20:16 UTC
  Pairs evaluated: 1098
  GT checksum (MD5): f504f38951b715aafb94cc4370f6385d
============================================================
  Metric            Mean        Std
  --------------    --------    --------
  PSNR              22.19 dB    3.26
  SSIM              0.714       0.064
  MS-SSIM           0.572       0.072
  LPIPS (AlexNet)   0.612       0.054
  LPIPS (VGG)       0.643       0.042
  MAE               0.064       0.029
  FID               274.53      --
  Runtime           --          --
  GPU Memory        --          --
============================================================
  Library versions
  --------------
  torch             2.1.0
  torchmetrics      1.4.0
  lpips             0.1.4
  torch-fidelity    0.3.0
  numpy             1.26.0
  Pillow            10.3.0
============================================================
  Saved to: results/results.csv
============================================================
```

With `--cellpose --cellpose_model cpsam --cellpose_n 200`:

```
============================================================
  ...same header and metrics table...
============================================================
  Cellpose (cpsam) -- 200 pairs sampled
  Metric            Mean        Std
  --------------    --------    --------
  CP Precision      0.823       0.041
  CP Recall         0.791       0.053
  CP F1             0.807       0.047
============================================================
  Library versions
  --------------
  torch             2.1.0
  ...
  cellpose          3.0.10
============================================================
```

---

## CSV output

Each run appends one row. The file is created with a header if it does not exist.
All floats are rounded to 6 decimal places. Missing values are written as `N/A`.

Columns (in order):
```
timestamp_utc, model_name, dataset_name, split_name, num_pairs,
psnr_mean, psnr_std,
ssim_mean, ssim_std,
ms_ssim_mean, ms_ssim_std,
lpips_alex_mean, lpips_alex_std,
lpips_vgg_mean, lpips_vgg_std,
mae_mean, mae_std,
fid,
runtime_s, gpu_mem_mb,
seed,
gt_checksum_md5,
torch_version, torchmetrics_version, lpips_version,
torch_fidelity_version, numpy_version, pillow_version,
cellpose_model, cellpose_n_pairs,
cellpose_precision_mean, cellpose_precision_std,
cellpose_recall_mean, cellpose_recall_std,
cellpose_f1_mean, cellpose_f1_std,
cellpose_version
```

Runs without `--cellpose` write `N/A` for all `cellpose_*` columns. This means you can
freely mix Cellpose and non-Cellpose runs in the same CSV and the file stays valid.

---

## Reproducibility

Every run logs:
- **GT checksum (MD5)**: proves the test set was identical across runs
- **UTC timestamp**: when the evaluation was run
- **Library versions**: exact versions of every metric library
- **Random seed**: set at startup for `random`, `numpy`, and `torch`

---

## Image format support

- Accepted extensions: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
- Hidden files (starting with `.`) are skipped
- RGBA and grayscale images are converted to RGB automatically
- **16-bit images** (microscopy TIFFs): detected automatically and normalised by 65535
  instead of 255. A warning is logged when this path is taken.
- If a predicted image has a different size than its ground truth pair, evaluation fails.
  All benchmark inference jobs should write 1024x1024 outputs so metrics are comparable.

---

## Training curve visualisation

`plot_training_curves.py` parses SLURM `.out` log files (or `loss_log.txt` files) and
produces a per-model training curve plot and a summary CSV of final-epoch loss averages.

Run it after training completes, before or alongside `evaluate.py`:

```bash
python plot_training_curves.py \
    --logs path/to/run_BCI.out path/to/run_MIST.out \
    --labels BCI MIST-HER2 \
    --name <ModelName> \
    --out-dir results/<ModelName>
```

Outputs written to `--out-dir`:

- `<name>_training_curves.png` : loss, LR, and wall-clock timing panels
- `<name>_training_summary.csv` : final-epoch averages per run; append these
  columns to the main benchmark table for reporting

Works with any model whose training script inherits the junyanz logger (CUT,
pix2pix, CycleGAN, PSPStain, ASP, and similar GAN repos). Restoration models
(SwinIR, NAFNet, etc.) may use a different log format : verify before use.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--logs` | required | One or more `.out` or `loss_log.txt` files |
| `--labels` | file stems | Display names matching `--logs` order |
| `--name` | `training` | Base name for output files |
| `--out-dir` | `.` | Output directory, created if absent |
| `--smooth` | `5` | Moving-average window for loss curves |
| `--last-n` | `10` | Final N epochs used for summary averages; `0` = all |
| `--dpi` | `150` | Output image DPI |

### On the cluster (VSC)

Use the wrapper `run_plot_training.sh` on the **login node** : it loads the correct
modules and activates the evaluation venv automatically:

```bash
bash run_plot_training.sh \
    --logs path/to/run_BCI.out path/to/run_MIST.out \
    --labels BCI MIST-HER2 \
    --name CUT \
    --out-dir $VSC_DATA/evaluate/results/CUT
```

No `sbatch` needed. Parsing and plotting are fast enough to run interactively.

---
