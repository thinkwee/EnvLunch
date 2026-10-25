"""Pack task pools into one shuffled slime training parquet."""

from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from envlunch.env import load_config, make_env


def main():
    """Read task jsonl files, render prompts, optionally subsample each pool, write parquet."""
    ap = argparse.ArgumentParser()
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-pool", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = load_config(args.config)
    rng = random.Random(args.seed)
    rows = []
    for path in args.pools:
        with open(path) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        if args.per_pool:
            tasks = rng.sample(tasks, min(args.per_pool, len(tasks)))
        for task in tasks:
            rows.append({"input": make_env(task, config).user_message(), "label": "",
                         "metadata": {"task_type": task["task_type"], "task": json.dumps(task)}})
    rng.shuffle(rows)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
