#!/bin/bash
# run_plot_training.sh
# Wrapper for plot_training_curves.py.
# Loads the evaluation environment and forwards all arguments to the script.
#
# Run on the LOGIN NODE only -- no sbatch needed.
#
# Usage (same flags as plot_training_curves.py):
#   bash run_plot_training.sh \
#       --logs path/to/run_BCI.out path/to/run_MIST.out \
#       --labels BCI MIST-HER2 \
#       --name CUT \
#       --out-dir $VSC_DATA/evaluate/results/CUT
#
# All flags are optional except --logs. See plot_training_curves.py --help.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_SCRIPT="$SCRIPT_DIR/plot_training_curves.py"
VENV_DIR="$VSC_DATA/evaluate/venv_eval"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# Run -- all arguments forwarded as-is
# ---------------------------------------------------------------------------

python "$PLOT_SCRIPT" "$@"

deactivate