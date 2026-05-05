# Evaluate — Cluster Setup and Usage Plan (Container Approach)

Complete reference for setting up and using the evaluate repo on VSC Tier 2 Antwerp
using an Apptainer container. This document describes the container-based workflow
introduced after the venv-based approach (see `cluster_plan.md`). Both approaches
remain valid -- this one is preferred for all new model evaluations.

Run all commands from the login node unless stated otherwise.

This repo is **shared benchmark infrastructure**. The container is built once locally,
uploaded once to scratch, and every model's eval sbatch script calls the same
`evaluate.py` through the same container image.
Each model team writes their own eval sbatch script -- they do not modify anything
in this repo.

---

## Why containers instead of a venv

The venv approach (see `cluster_plan.md`) is tied to the fixed PyTorch module stack on
the cluster (`PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1`). The container approach
is independent of that module stack. The container carries its own Python, PyTorch,
and all evaluation dependencies. Only `calcua/2026.1` needs to be loaded on the host
to make the `apptainer` command available.

| Aspect               | Venv approach              | Container approach                        |
|----------------------|----------------------------|-------------------------------------------|
| Python / PyTorch     | Fixed to module stack      | Bundled in container (2.7.0 / cu128)      |
| Setup on cluster     | Run install_eval.sh        | Upload .sif, done                         |
| Module loads in jobs | calcua + SciPy + PyTorch   | calcua/2026.1 only                        |
| Environment location | $VSC_DATA/evaluate/venv_eval/ | $VSC_SCRATCH/containers/evaluate_nvidia.sif |

---

## What this repo contains

| Path                                  | What it is                                                                             |
|---------------------------------------|----------------------------------------------------------------------------------------|
| `evaluate.py`                         | The benchmark evaluation script. Reads two image folders, computes metrics, writes CSV |
| `hpc_jobs/evaluate_nvidia.def`        | Apptainer container definition. Build this locally to produce the .sif image           |
| `hpc_jobs/cluster_plan.md`            | Setup guide for the venv-based approach                                                |
| `hpc_jobs/cluster_plan_container.md`  | This file                                                                              |
| `environment.yml`                     | Conda env spec for local development only -- not used on the cluster                   |
| `README.md`                           | Full argument reference, metric descriptions, output format docs                       |

The `.sif` file is never committed to git. It is built locally from `evaluate_nvidia.def`
and uploaded to `$VSC_SCRATCH/containers/`.

---

## Prerequisites

| Requirement                        | Provided by                                               |
|------------------------------------|-----------------------------------------------------------|
| Cluster account and SSH access     | VSC onboarding                                            |
| `$VSC_DATA` and `$VSC_SCRATCH`     | Cluster environment (automatic)                           |
| Apptainer on local Linux machine   | `sudo dnf install apptainer` (Fedora) or via PPA (Ubuntu) |
| Model predictions on disk          | The model's own inference sbatch job                      |
| Ground truth images via sqsh mount | Dataset squashfs archive on scratch                       |

`evaluate.py` never touches training code, checkpoints, or raw datasets. It only reads
two folders of images.

---

## A note on internet access

The login node has full internet access. Compute nodes have no outbound internet.
LPIPS and Cellpose both download weights from external URLs on first use. These weights
must be pre-downloaded on the login node (Step B3 below) so compute jobs find them
cached in `~/.cache` and `~/.cellpose/models/` without any network access.

---

## One-time setup

### Step B1 -- Build the container locally

Run on your local Linux machine from the `hpc_jobs/` directory:

```bash
cd ~/projects/evaluate/hpc_jobs

sudo APPTAINER_TMPDIR=$HOME/apptainer_tmp APPTAINER_CACHEDIR=$HOME/apptainer_cache \
    apptainer build evaluate_nvidia.sif evaluate_nvidia.def
```

The build takes 10 to 20 minutes. Expected last line: **INFO:    Build complete: evaluate_nvidia.sif**

Verify non-GPU imports locally after build:
```bash
apptainer exec evaluate_nvidia.sif python -c "
from PIL import Image; print('Pillow ok')
import numpy as np; print('numpy:', np.__version__)
import torchmetrics; print('torchmetrics:', torchmetrics.__version__)
import torch_fidelity; print('torch-fidelity ok')
import importlib.metadata; print('cellpose:', importlib.metadata.version('cellpose'))
print('Non-GPU imports OK')
"
```

### Step B2 -- Clone the evaluate repo and upload the container

Clone the repo to the cluster (run once):
```bash
git clone https://github.com/InViLabVirtualStainingBenchmark/evaluate.git $VSC_DATA/evaluate
```

Create the containers directory on scratch (run once):
```bash
ssh vsc[YOUR_USER]@login.hpc.uantwerpen.be "mkdir -p $VSC_SCRATCH/containers"
```

Upload the container (use rsync to allow resuming if interrupted):
```bash
rsync -avz --progress evaluate_nvidia.sif \
    vsc[YOUR_USER]@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/evaluate_nvidia.sif
```

Verify on the cluster:
```bash
ls -lh $VSC_SCRATCH/containers/
```

### Step B3 -- Full import test and weight pre-download (login node)

SSH into the cluster and load the module:
```bash
module purge
module load calcua/2026.1
```

Run the full import test with `--nv`:
```bash
apptainer exec --nv $VSC_SCRATCH/containers/evaluate_nvidia.sif python -c "
import torch; print('torch:', torch.__version__)
import torchmetrics; print('torchmetrics:', torchmetrics.__version__)
import lpips; print('lpips ok')
import torch_fidelity; print('torch-fidelity ok')
import importlib.metadata; print('cellpose:', importlib.metadata.version('cellpose'))
print('cuda available:', torch.cuda.is_available())
print('All imports OK')
"
```

`cuda available: False` on the login node is expected.

Check whether LPIPS weights are already cached:
```bash
ls ~/.cache/torch/hub/checkpoints/
```

Expected: `alexnet-owt-*.pth` and `vgg16-*.pth` present. If missing, download them:
```bash
apptainer exec --nv $VSC_SCRATCH/containers/evaluate_nvidia.sif python -c "
import lpips
lpips.LPIPS(net='alex')
lpips.LPIPS(net='vgg')
print('LPIPS weights cached.')
"
```

Check whether Cellpose weights are cached:
```bash
ls ~/.cellpose/models/
```

Expected: `cpsam` present (Cellpose 4.x default model). If missing, download it:
```bash
apptainer exec --nv $VSC_SCRATCH/containers/evaluate_nvidia.sif python -c "
from cellpose import models
models.CellposeModel(pretrained_model='cpsam')
print('cpsam cached.')
"
```

**Gate:** all imports pass and both weight directories are populated before submitting
any eval job for any model.

---

## How model eval jobs use this repo

Each model has its own eval sbatch script in that model's `hpc_jobs/` folder. The
script mounts the model's output directory and the dataset squashfs archive into
the evaluate container and calls `evaluate.py` directly.

The pattern all container-based eval scripts follow:

```bash
#!/bin/bash
#SBATCH --job-name={model}_eval_{DATASET}
#SBATCH --account=ap_invilab_td_thesis
#SBATCH --partition=ampere_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
PRED_DIR="$VSC_DATA/projects/{model}/outputs/results/{run}/test_output"
GT_DIR="/data/{DATASET}/IHC/test"
OUTPUT_CSV="$VSC_DATA/benchmark_results.csv"
EVAL_SCRIPT="$VSC_DATA/evaluate/evaluate.py"

module purge
module load calcua/2026.1

# Pre-flight checks
for path in "$PRED_DIR" "$EVAL_SCRIPT"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: required path not found: $path"
        exit 1
    fi
done

srun apptainer exec --nv \
    -B $VSC_DATA:$VSC_DATA \
    -B $VSC_SCRATCH/BCI.sqsh:/data/BCI:image-src=/ \
    $CONTAINER python "$EVAL_SCRIPT" \
        --pred    "$PRED_DIR" \
        --gt      "$GT_DIR" \
        --model_name   {model} \
        --dataset_name {DATASET} \
        --split_name   test \
        --match_by     stem \
        --output       "$OUTPUT_CSV" \
        --device       cuda
```

`PRED_DIR`, `--model_name`, and `--dataset_name` are the only model-specific variables.
All results append to the same `$VSC_DATA/benchmark_results.csv`.

For unpaired models use `--match_by sort` instead of `--match_by stem`.

### Adding Cellpose to an eval script

Add these arguments to the `python evaluate.py` call:

```bash
    --cellpose \
    --cellpose_model cpsam \
    --cellpose_n 200
```

`--cellpose_n 200` limits Cellpose to 200 randomly sampled pairs (seeded, reproducible).

---

## Rebuilding the container

If a dependency needs to be updated, edit `evaluate_nvidia.def` locally and rebuild:

```bash
cd ~/projects/evaluate/hpc_jobs
sudo APPTAINER_TMPDIR=$HOME/apptainer_tmp APPTAINER_CACHEDIR=$HOME/apptainer_cache \
    apptainer build evaluate_nvidia.sif evaluate_nvidia.def
rsync -avz --progress evaluate_nvidia.sif \
    vsc21212@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/evaluate_nvidia.sif
```

Commit the updated `.def` file. Do not commit the `.sif` file.

## Updating evaluate.py

```bash
cd $VSC_DATA/evaluate
git pull
```

No container rebuild needed if only `evaluate.py` changed.

---

## Key paths

| Artifact               | Path                                                   |
|------------------------|--------------------------------------------------------|
| Container definition   | `~/projects/evaluate/hpc_jobs/evaluate_nvidia.def`     |
| Container image        | `$VSC_SCRATCH/containers/evaluate_nvidia.sif`          |
| Evaluate repo          | `$VSC_DATA/evaluate/`                                  |
| evaluate.py            | `$VSC_DATA/evaluate/evaluate.py`                       |
| Shared results CSV     | `$VSC_DATA/benchmark_results.csv`                      |
| LPIPS weight cache     | `~/.cache/torch/hub/checkpoints/`                      |
| Cellpose weight cache  | `~/.cellpose/models/`                                  |

---

## Common issues

| Problem                                                | Cause                                                               | Fix                                                                                        |
|--------------------------------------------------------|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `libcusparseLt.so.0: cannot open shared object file`   | Login node lacks this CUDA lib; must come from container base image | Ensure base image is `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime`, not a raw CUDA image |
| `apptainer: command not found`                         | calcua module not loaded                                            | Run `module load calcua/2026.1` first                                                      |
| File not found inside container                        | VSC_DATA or VSC_SCRATCH not bound                                   | Add `-B $VSC_DATA:$VSC_DATA -B $VSC_SCRATCH:$VSC_SCRATCH` to the apptainer exec call       |
| Dataset path not found inside container                | sqsh mount path mismatch                                            | Verify the `-B *.sqsh:/data/...:image-src=/` path matches what `--gt` expects              |
| LPIPS or Cellpose weight download fails in compute job | Weights not pre-downloaded on login node                            | Run Step B3 on the login node before submitting any eval job                               |
| `cuda available: False` on login node                  | No GPU on login node                                                | Expected -- GPU is available on compute nodes only                                         |
| Container build runs out of /tmp space                 | /tmp too small for CUDA base image                                  | Set `APPTAINER_TMPDIR=$HOME/apptainer_tmp` before building                                 |