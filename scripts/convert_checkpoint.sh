#!/usr/bin/env bash
# Convert a Hugging Face checkpoint to Megatron torch_dist.
set -euo pipefail
: "${HF_MODEL:?set HF_MODEL}"
: "${TORCH_DIST:?set TORCH_DIST (output directory)}"
: "${MODEL_ARCH:?set MODEL_ARCH, e.g. configs/qwen3.5-4B.sh}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SLIME_DIR=${SLIME_DIR:-$ROOT/third_party/slime}
MEGATRON_DIR=${MEGATRON_DIR:-$ROOT/third_party/Megatron-LM}
source "$ROOT/$MODEL_ARCH"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT:-29680} RANK=0 WORLD_SIZE=1 LOCAL_RANK=0

cd "$SLIME_DIR"
PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR" python3 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" --hf-checkpoint "$HF_MODEL" --save "$TORCH_DIST"
