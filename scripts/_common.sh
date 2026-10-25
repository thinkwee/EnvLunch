#!/usr/bin/env bash
# Shared slime launch settings (GRPO, optimiser, parallelism, SGLang) and the launch() helper.
set -euo pipefail

: "${HF_MODEL:?set HF_MODEL to the Hugging Face checkpoint}"
: "${TORCH_DIST:?set TORCH_DIST to the Megatron torch_dist checkpoint}"
: "${MODEL_ARCH:?set MODEL_ARCH, e.g. configs/qwen3.5-4B.sh}"
: "${TRAIN_PARQUET:?set TRAIN_PARQUET}"
: "${SAVE_DIR:?set SAVE_DIR}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SLIME_DIR=${SLIME_DIR:-$ROOT/third_party/slime}
MEGATRON_DIR=${MEGATRON_DIR:-$ROOT/third_party/Megatron-LM}
source "$ROOT/$MODEL_ARCH"

ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl
    --entropy-coef 0.001
    --eps-clip 0.2 --eps-clip-high 0.28
    --use-tis --tis-clip 1.5 --tis-clip-low 0.5
)
OPTIMIZER_ARGS=(
    --optimizer adam --lr 1e-6 --lr-decay-style constant
    --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.98
)
PERF_ARGS=(
    --tensor-model-parallel-size "$TP_SIZE" --pipeline-model-parallel-size 1 --sequence-parallel
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
    --use-distributed-optimizer
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.8}"
    --sglang-chunked-prefill-size 4096
    --sglang-server-concurrency "${SGLANG_CONCURRENCY:-32}"
)
MISC_ARGS=(
    --attention-dropout 0.0 --hidden-dropout 0.0
    --attention-softmax-in-fp32 --attention-backend flash
)
CKPT_ARGS=(
    --hf-checkpoint "$HF_MODEL" --ref-load "$TORCH_DIST"
    --load "$SAVE_DIR" --save "$SAVE_DIR"
    --no-load-optim --no-save-optim
)
DATA_ARGS=(
    --prompt-data "$TRAIN_PARQUET" --input-key input --label-key label --metadata-key metadata
    --rollout-temperature 1.0 --rollout-max-response-len "$MAX_RESPONSE_LEN"
    --data-source-path envlunch.data_source.TextOnlyRolloutDataSource
)

export ENVLUNCH_CONFIG=${ENVLUNCH_CONFIG:-$ROOT/configs/warmup.yaml}
RUNTIME_ENV=$(PYTHONPATH="${ROOT}:${MEGATRON_DIR}:${SLIME_DIR}" SLIME_SAVE_AT="${SLIME_SAVE_AT:-}" python3 -c '
import json, os
keep = ("PYTHONPATH", "SLIME_SAVE_AT")
env = {k: v for k, v in os.environ.items() if k in keep or k.startswith("ENVLUNCH_")}
env.update({"CUDA_DEVICE_MAX_CONNECTIONS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SGLANG_DISABLE_CUDNN_CHECK": "1"})
print(json.dumps({"env_vars": env}))')

launch() {
    local rc=0
    ray start --head --num-gpus $((ACTOR_GPUS + ROLLOUT_GPUS)) --disable-usage-stats
    ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="$RUNTIME_ENV" -- \
        python3 "$SLIME_DIR/train_async.py" \
        --actor-num-nodes 1 --actor-num-gpus-per-node "$ACTOR_GPUS" --rollout-num-gpus "$ROLLOUT_GPUS" \
        "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${DATA_ARGS[@]}" "$@" \
        "${OPTIMIZER_ARGS[@]}" "${GRPO_ARGS[@]}" "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" || rc=$?
    ray stop
    return $rc
}
