#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: please activate your conda environment first."
  echo "Example: conda activate object-tracking-eval"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/3] Setting up compiler environment for PyTorch 1.13.1 + CUDA 11.7..."

# Export environment variables for compilation
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6}"

# Use conda's compilers
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"

# Add conda lib to library path
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "[2/3] Verifying CUDA environment..."
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
assert torch.cuda.is_available(), "CUDA must be available"
assert CUDA_HOME is not None, "CUDA_HOME must be set"
PY

echo "[3/3] Building RCBEVDet CUDA ops..."

# Clean previous builds
rm -rf build
rm -rf src/model/det/rcbevdet/ops/deformattn/build
rm -rf src/model/det/rcbevdet/ops/bev_pool_v2/build

# Build deformable attention
cd src/model/det/rcbevdet/ops/deformattn
python setup.py build_ext --inplace 2>&1 | tee compile_deformattn.log
cd "$ROOT_DIR"

# Build BEV pool v2
cd src/model/det/rcbevdet/ops/bev_pool_v2
python setup.py build_ext --inplace 2>&1 | tee compile_bev_pool.log
cd "$ROOT_DIR"

echo ""
echo "================================================"
echo "Compilation complete!"
echo "================================================"
echo ""
echo "Check logs if there were errors:"
echo "  - src/model/det/rcbevdet/ops/deformattn/compile_deformattn.log"
echo "  - src/model/det/rcbevdet/ops/bev_pool_v2/compile_bev_pool.log"
