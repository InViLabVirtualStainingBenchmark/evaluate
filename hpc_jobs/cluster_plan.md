# Evaluate — Cluster Setup and Usage Plan

Complete reference for setting up and using the evaluate repo on VSC Tier 2 Antwerp.
Run all commands from the login node unless stated otherwise.

This repo is **shared benchmark infrastructure**. It is cloned once to `$VSC_DATA/evaluate/`
and every model's eval sbatch script sources the same venv and calls the same `evaluate.py`.
Each model team writes their own eval sbatch script — they do not modify anything in this repo.

For cluster connection details, partition specs, and storage paths see the CUT execution
plan at https://github.com/InViLabVirtualStainingBenchmark/contrastive-unpaired-translation/blob/master/hpc_jobs/execution_plan.md.

---

## What this repo contains

| Path                       | What it is                                                                              |
|----------------------------|-----------------------------------------------------------------------------------------|
| `evaluate.py`              | The benchmark evaluation script. Reads two image folders, computes metrics, writes CSV  |
| `hpc_jobs/install_eval.sh` | Login-node bash script that builds the shared eval venv and pre-downloads model weights |
| `hpc_jobs/cluster_plan.md` | This file                                                                               |
| `environment.yml`          | Conda env spec for local development only — not used on the cluster                     |
| `README.md`                | Full argument reference, metric descriptions, output format docs                        |

---

## Prerequisites

| Requirement                       | Provided by                                          |
|-----------------------------------|------------------------------------------------------|
| Cluster account and SSH access    | VSC onboarding                                       |
| `$VSC_DATA` and `$VSC_SCRATCH`    | Cluster environment (automatic)                      |
| Model predictions on disk         | The model's own inference sbatch job                 |
| Ground truth images on disk       | Dataset transfer (see model-specific execution plan) |

`evaluate.py` never touches training code, checkpoints, or raw datasets. It only reads
two folders of images.

---

## A note on internet access

The login node has full internet access. Compute nodes can reach PyPI (so `pip install`
inside a sbatch job works) but cannot reach arbitrary external URLs such as HuggingFace
model hubs or GitHub release assets. LPIPS and Cellpose both download weights from such
URLs on first use. `install_eval.sh` is therefore run on the **login node**, not submitted
as a batch job — that is the only way to guarantee internet access for the weight downloads.

---

## Cluster setup (one-time)

### Step A1 — Clone the evaluate repo

```bash
git clone https://github.com/InViLabVirtualStainingBenchmark/evaluate.git $VSC_DATA/evaluate
```

Verify:
```bash
ls $VSC_DATA/evaluate
```
Expected: `evaluate.py  environment.yml  hpc_jobs  README.md`

```bash
ls $VSC_DATA/evaluate/hpc_jobs
```
Expected: `cluster_plan.md  install_eval.sh`

---

## Environment setup (login node, run once)

Run `install_eval.sh` directly on the login node. It is **not** submitted via `sbatch`.

```bash
bash $VSC_DATA/evaluate/hpc_jobs/install_eval.sh
```

**What the script does, in order:**
1. Loads the cluster PyTorch module stack
2. Wipes and recreates `$VSC_DATA/evaluate/venv_eval/` from scratch
3. pip-installs: `torchmetrics`, `lpips`, `torch-fidelity`, `cellpose`
4. Pre-downloads LPIPS backbone weights (AlexNet, VGG) into `~/.cache`
5. Pre-downloads Cellpose model weights into `~/.cellpose/models/` (cyto2 by default)
6. Runs sanity-check imports and prints all library versions

**Gate:** the script must end with `Eval venv install complete. All checks passed.`
before any eval sbatch job for any model is submitted. If a check fails, fix the
relevant block in `install_eval.sh` and re-run.

---

## How model eval jobs use this repo

`evaluate.py` is never called directly from the command line on the cluster. Each model
has its own eval sbatch script (in that model's `hpc_jobs/` folder) that sources the
shared venv and calls `evaluate.py`. See the CUT eval scripts as concrete examples:
https://github.com/InViLabVirtualStainingBenchmark/contrastive-unpaired-translation/tree/master/hpc_jobs

The pattern all eval scripts follow:

```bash
#!/bin/bash
#SBATCH ...resource directives...

PRED_DIR="<path to model inference output>"
GT_DIR="<path to ground truth test images>"
OUTPUT_CSV="$VSC_DATA/benchmark_results.csv"
EVAL_SCRIPT="$VSC_DATA/evaluate/evaluate.py"
VENV_DIR="$VSC_DATA/evaluate/venv_eval"

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VENV_DIR/bin/activate"

# --- pre-flight checks: verify PRED_DIR, GT_DIR, and EVAL_SCRIPT exist ---

python "$EVAL_SCRIPT" \
    --pred         "$PRED_DIR" \
    --gt           "$GT_DIR" \
    --model_name   <model_label> \
    --dataset_name <dataset_label> \
    --split_name   test \
    --match_by     stem \
    --output       "$OUTPUT_CSV" \
    --device       cuda

deactivate
```

`PRED_DIR`, `--model_name`, and `--dataset_name` are the only model-specific variables.
All results append to the same `$VSC_DATA/benchmark_results.csv`.

### Adding Cellpose to an eval script

Add three arguments to the `python evaluate.py` call in the model's eval script:

```bash
    --cellpose \
    --cellpose_model cyto2 \
    --cellpose_n 200
```

`--cellpose_n 200` limits Cellpose to 200 randomly sampled pairs (seeded, reproducible).
Omit it to run on all pairs, which is slower but gives lower-variance estimates.

| `--cellpose_model` | Use for                                       |
|--------------------|-----------------------------------------------|
| `cyto2` (default)  | H&E staining, cytoplasm                       |
| `cyto3`            | Cytoplasm, newer — more accurate, more memory |
| `nuclei`           | DAPI, Hoechst, or other nuclear stains        |

---

## Adding a new Cellpose model weight

If you need a Cellpose model not yet cached (e.g. `nuclei`):

1. Open `$VSC_DATA/evaluate/hpc_jobs/install_eval.sh` on the login node
2. Uncomment the relevant `CellposeModel(pretrained_model=...)` line
3. Re-run: `bash $VSC_DATA/evaluate/hpc_jobs/install_eval.sh`

On re-run the venv is rebuilt and `cyto2` is already cached, so only the new model downloads.

---

## Updating evaluate.py or install_eval.sh

```bash
cd $VSC_DATA/evaluate
git pull
```

If `install_eval.sh` changed (new dependency or pip version bump), re-run it on the login node:

```bash
bash $VSC_DATA/evaluate/hpc_jobs/install_eval.sh
```

If only `evaluate.py` changed, no reinstall is needed.

---

## Key paths

| Artifact              | Path                                          |
|-----------------------|-----------------------------------------------|
| Evaluate repo         | `$VSC_DATA/evaluate/`                         |
| evaluate.py           | `$VSC_DATA/evaluate/evaluate.py`              |
| install_eval.sh       | `$VSC_DATA/evaluate/hpc_jobs/install_eval.sh` |
| Shared eval venv      | `$VSC_DATA/evaluate/venv_eval/`               |
| Shared results CSV    | `$VSC_DATA/benchmark_results.csv`             |
| LPIPS weight cache    | `~/.cache/torch/hub/`                         |
| Cellpose weight cache | `~/.cellpose/models/`                         |

---

## Common issues

| Problem                                                                | Cause                                         | Fix                                                                                                  |
|------------------------------------------------------------------------|-----------------------------------------------|------------------------------------------------------------------------------------------------------|
| `module: command not found` when running install_eval.sh               | Script run outside the cluster                | Must be run on the VSC login node, not locally                                                       |
| pip install fails                                                      | PyPI unreachable — unlikely on VSC login node | Check your network connection; retry                                                                 |
| Cellpose weight download fails                                         | HuggingFace unreachable                       | Check login node internet: `curl -I https://huggingface.co`. If blocked, contact VSC support.        |
| `ModuleNotFoundError` in eval sbatch job                               | Wrong venv sourced or install not completed   | Confirm `source .../venv_eval/bin/activate` runs before `python evaluate.py`; recheck install output |
| CSV has `N/A` in all `cellpose_*` columns                              | `--cellpose` not in the model's eval script   | Add `--cellpose` (and `--cellpose_model`, `--cellpose_n`) to the eval sbatch script's python call    |
| `AttributeError: module 'cellpose.models' has no attribute 'Cellpose'` | Old evaluate.py before Cellpose 4.x fix       | `git pull` in `$VSC_DATA/evaluate` to get the current version                                        |
| CSV missing header row                                                 | File existed from a partial previous run      | Delete `benchmark_results.csv` and rerun — header is written only when the file does not exist       |
