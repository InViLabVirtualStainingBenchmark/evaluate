#!/bin/bash
# install_eval.sh
# Historical name kept for compatibility. Evaluation now uses the shared
# evaluate_nvidia.sif container, so there is no cluster venv installation step.
#
# Run this on the login node to verify artifacts and pre-download weights:
#   bash hpc_jobs/install_eval.sh

set -euo pipefail

EVAL_CONTAINER="$VSC_SCRATCH/containers/evaluate_nvidia.sif"
EVAL_SCRIPT="$VSC_DATA/evaluate/evaluate.py"

module purge
module load calcua/2026.1

echo "=== Shared evaluation container ==="
if [ ! -f "$EVAL_CONTAINER" ]; then
    echo "ERROR: evaluate_nvidia.sif not found: $EVAL_CONTAINER"
    echo "Build it locally from hpc_jobs/evaluate_nvidia.def and upload it to $VSC_SCRATCH/containers/."
    exit 1
fi
echo "  found: $EVAL_CONTAINER"

echo ""
echo "=== Cluster evaluate.py ==="
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: evaluate.py not found: $EVAL_SCRIPT"
    echo "Clone or update the evaluate repo at $VSC_DATA/evaluate."
    exit 1
fi
echo "  found: $EVAL_SCRIPT"

echo ""
echo "=== Import check and weight pre-download ==="
apptainer exec --nv \
    -B "$VSC_DATA:$VSC_DATA" \
    "$EVAL_CONTAINER" \
    python - <<'PY'
import importlib.metadata
import os
import tempfile

import lpips
import numpy as np
from PIL import Image
import torch
import torch_fidelity
import torchmetrics

print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("torchmetrics:", importlib.metadata.version("torchmetrics"))
print("lpips:", importlib.metadata.version("lpips"))
print("torch-fidelity:", importlib.metadata.version("torch-fidelity"))
try:
    print("cellpose:", importlib.metadata.version("cellpose"))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit("ERROR: cellpose is not installed in evaluate_nvidia.sif")

print("")
print("Pre-downloading LPIPS weights...")
lpips.LPIPS(net="alex")
lpips.LPIPS(net="vgg")
print("LPIPS weights cached.")

print("")
print("Pre-downloading torch-fidelity Inception weights...")
with tempfile.TemporaryDirectory() as tmp:
    pred_dir = os.path.join(tmp, "pred")
    gt_dir = os.path.join(tmp, "gt")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for i in range(4):
        arr = np.full((64, 64, 3), i * 40, dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(pred_dir, f"{i:03d}.png"))
        Image.fromarray(255 - arr).save(os.path.join(gt_dir, f"{i:03d}.png"))

    torch_fidelity.calculate_metrics(
        input1=pred_dir,
        input2=gt_dir,
        cuda=False,
        isc=False,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
print("torch-fidelity weights cached.")

print("")
print("Pre-downloading Cellpose cpsam weights...")
from cellpose import models as cellpose_models
cellpose_models.CellposeModel(gpu=False, pretrained_model="cpsam")

# Uncomment and re-run this script if an eval job switches to another model.
# cellpose_models.CellposeModel(gpu=False, pretrained_model="cyto2")
# cellpose_models.CellposeModel(gpu=False, pretrained_model="cyto3")
# cellpose_models.CellposeModel(gpu=False, pretrained_model="nuclei")
# cellpose_models.CellposeModel(gpu=False, pretrained_model="cyto")
print("Cellpose cpsam weights cached.")
PY

echo ""
echo "=== Cache locations ==="
echo "Compute nodes have no internet. These caches were warmed on the login node:"
echo "  ~/.cache/torch/hub/checkpoints/      LPIPS and torch-fidelity weights"
echo "  ~/.cellpose/models/cpsam             Cellpose cpsam weights"

echo ""
echo "Evaluation infrastructure check complete."
