# EnvLunch

Code for *[EnvLunch: The Environment as a Free Lunch for Learning to Interact](https://arxiv.org/abs/2610.33817)*.

## Setup

```bash
# Install slime, Megatron-LM and SGLang at the versions used in the paper
bash scripts/setup_env.sh
```

## Data

```bash
# Download the raw sources: Spider, BIRD train, DataMind tables, DABench tables (~20 GB)
bash scripts/download_data.sh data/raw

# Turn the downloads into SQL databases and dataframe environments
python -m envlunch.prepare --raw data/raw --out data/envs

# Synthesise SQL tasks: random operator compositions, kept only if the result changes on perturbed database copies
python -m envlunch.sql.build_pool --db-glob "data/envs/sql/*.sqlite" --copy-dir data/copies --out data/sql.jsonl

# Synthesise dataframe tasks in the same way
python -m envlunch.frame.build_pool --spec-glob "data/envs/frame/*" --out data/frame.jsonl

# Mix 2451 tasks from each pool into the training file
python -m envlunch.make_parquet data/sql.jsonl data/frame.jsonl --per-pool 2451 --out data/warmup.parquet
```

## Training

```bash
# Model to warm up (use configs/qwen3.5-9B.sh and Qwen/Qwen3.5-9B for 9B)
export MODEL_ARCH=configs/qwen3.5-4B.sh
export HF_MODEL=Qwen/Qwen3.5-4B TORCH_DIST=ckpt/qwen3.5-4b-torch-dist

# Convert the Hugging Face checkpoint to Megatron format
bash scripts/convert_checkpoint.sh

# GRPO warm-up on 8 GPUs (4 training, 4 rollout); the final checkpoint is ckpt/warmup/iter_0000099
TRAIN_PARQUET=data/warmup.parquet SAVE_DIR=ckpt/warmup bash scripts/warmup.sh
```
