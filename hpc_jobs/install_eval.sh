#!/bin/bash

# install_eval.sh
# Run directly on the login node -- NOT submitted via sbatch.
# The login node has full internet access, which is required to pip-install
# packages and to pre-download model weights (HuggingFace, etc.).
# Compute nodes cannot reach those external URLs, so all caching must happen here.
#
# Creates the shared evaluation venv at $VSC_DATA/evaluate/venv_eval/.
# This venv is reused by all model eval jobs across the benchmark.
# Run once. Re-run only if you need to update a dependency or add a new
# Cellpose model weight.
#
# Run:  bash $VSC_DATA/evaluate/hpc_jobs/install_eval.sh
# Gate: script must end with "Eval venv install complete. All checks passed."
#       before submitting any eval job for any model.

set -euo pipefail

EVAL_DIR="$VSC_DATA/evaluate"
VENV_DIR="$EVAL_DIR/venv_eval"

# =========================
# MODULES
# =========================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

echo "=== System Python ==="
which python
python -V

# =========================
# CREATE VENV
# =========================

rm -rf "$VENV_DIR"
python -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"

echo ""
echo "=== Venv Python ==="
which python
python -V

python -m pip install --upgrade pip

# =========================
# EVALUATION DEPENDENCIES
# =========================

python -m pip install \
    torchmetrics \
    lpips \
    torch-fidelity \
    cellpose \
    --no-cache-dir

# =========================
# PRE-DOWNLOAD LPIPS WEIGHTS
# Compute nodes have no internet. Downloading here caches the weights
# in ~/.cache so all future eval jobs find them without network access.
# =========================

echo ""
echo "=== Pre-downloading LPIPS backbone weights ==="
python -c "
import lpips
print('Downloading AlexNet backbone...')
lpips.LPIPS(net='alex')
print('Downloading VGG backbone...')
lpips.LPIPS(net='vgg')
print('LPIPS weights cached.')
"

# =========================
# PRE-DOWNLOAD CELLPOSE WEIGHTS
# Compute nodes have no internet. Cellpose caches weights in ~/.cellpose/models/.
# Running CellposeModel() here on the login node (which has internet) pre-fills
# that cache. On re-runs, Cellpose skips any model whose file already exists,
# so only newly uncommented models are actually downloaded.
#
# To add a model for a new dataset: uncomment the relevant line(s) and
# re-run this script. Only the new weight(s) will be fetched.
#
# Model reference:
#   cyto2  -- 2nd-gen cytoplasm model. Default. Best for H&E and general
#              cytoplasmic staining. Use this unless you have a reason not to.
#   cyto3  -- 3rd-gen cytoplasm model. Newer and generally more accurate than
#              cyto2, but uses more memory. Good alternative if cyto2 undersegments.
#   nuclei -- Nuclear segmentation. Use for DAPI, Hoechst, or any stain where
#              you want to count/compare nuclei rather than whole cells.
#   cyto   -- Original (v1) cytoplasm model. Kept for reproducibility with older
#              results. Prefer cyto2 or cyto3 for new experiments.
# =========================

echo ""
echo "=== Pre-downloading Cellpose model weights ==="
python -c "
from cellpose import models

print('Downloading cyto2 weights...')
models.CellposeModel(pretrained_model='cyto2')

# Uncomment to pre-download additional models:
# print('Downloading cyto3 weights...')
# models.CellposeModel(pretrained_model='cyto3')

# print('Downloading nuclei weights...')
# models.CellposeModel(pretrained_model='nuclei')

# print('Downloading cyto (v1) weights...')
# models.CellposeModel(pretrained_model='cyto')

print('Cellpose weights cached.')
"

# =========================
# SANITY CHECKS
# =========================

echo ""
echo "=== Sanity checks ==="
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torchmetrics; print('torchmetrics:', torchmetrics.__version__)"
python -c "import lpips; print('lpips ok')"
python -c "import torch_fidelity; print('torch-fidelity ok')"
python -c "import cellpose; print('cellpose:', cellpose.__version__)"

deactivate
echo ""
echo "Eval venv install complete. All checks passed."
echo "Shared eval venv is at: $VENV_DIR"
echo "All model eval jobs should source this venv."
