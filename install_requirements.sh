#!/bin/bash
# =============================================================================
# Environment setup for ERGM_XOM.
#
# Creates a conda environment named `ergm_xom` and installs PyTorch + DGL +
# PyTorch Geometric matching the versions this codebase was developed and
# last verified against, followed by everything else in requirements.txt.
#
# GPU/CUDA note: the pinned wheels below target CUDA 12.8 (cu128), matching
# the development environment. If your GPU driver supports a different CUDA
# version, edit the `CUDA_TAG` variable below before running — installing
# mismatched CUDA wheels is the single most common source of "works on
# import, segfaults/hangs on first .cuda() call" bugs with this stack.
# Check your driver's max supported CUDA version with `nvidia-smi` first.
#
# Usage:
#   bash install_requirements.sh
# =============================================================================
set -euo pipefail

ENV_NAME="SCR"
PYTHON_VERSION="3.11"
CUDA_TAG="cu128"          # change to e.g. cu121, cu118, or "cpu" if needed
TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first."
    exit 1
fi

echo "=== Creating conda environment '${ENV_NAME}' (python ${PYTHON_VERSION}) ==="
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"

# `conda activate` inside a non-interactive script requires sourcing conda's
# shell hook first (plain `source activate` from an interactive shell relies
# on shell init files this script doesn't have).
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch ${TORCH_VERSION} (${CUDA_TAG}) ==="
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=== Installing torch-scatter (matching torch ${TORCH_VERSION}/${CUDA_TAG}) ==="
pip install torch-scatter \
    -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "=== Installing torch-geometric ==="
pip install torch-geometric

echo "=== Installing DGL (${CUDA_TAG}) ==="
# DGL's own wheel index names CUDA tags without the leading "cu", e.g. "128".
DGL_CUDA_TAG="${CUDA_TAG#cu}"
if [ "${CUDA_TAG}" = "cpu" ]; then
    pip install dgl -f https://data.dgl.ai/wheels/repo.html
else
    pip install "dgl" -f "https://data.dgl.ai/wheels/cu${DGL_CUDA_TAG}/repo.html"
fi

echo "=== Installing remaining Python dependencies from requirements.txt ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "=== Done. Verify the install with: ==="
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"import torch, torch_geometric, dgl; print(torch.__version__, torch_geometric.__version__, dgl.__version__)\""
echo ""
echo "Note: ORCA (orbit-counting binary for the orbits_mmd metric) is NOT"
echo "installed by this script — see THIRD_PARTY_LICENSES.md and README.md"
echo "for why, and how to obtain it separately if you need that metric."
