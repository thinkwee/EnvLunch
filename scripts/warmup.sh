#!/usr/bin/env bash
# EnvLunch warm-up: 100 GRPO steps of 16 tasks x 8 samples on the warm-up parquet.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

launch \
    --num-rollout "${STEPS:-100}" \
    --save-interval "${SAVE_INTERVAL:-25}" \
    --rollout-batch-size "${ROLLOUT_BATCH:-16}" \
    --n-samples-per-prompt "${N_SAMPLES:-8}" \
    --global-batch-size "${GLOBAL_BATCH:-32}" \
    --balance-data \
    --custom-generate-function-path envlunch.rollout.generate \
    --custom-rm-path envlunch.rollout.reward_func \
    --group-rm
