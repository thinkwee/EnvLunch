"""Dataframe trajectory synthesis -> tasks.jsonl."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re

from envlunch.env import load_config
from envlunch.frame.actions import tasks_for_relations
from envlunch.frame.relations import load_relations


def target_repr(base) -> str:
    """Target output as text."""
    if isinstance(base, float):
        return f"{round(base, 4)}"
    try:
        return "{" + ", ".join(f"{k}: {round(v, 4)}" for k, v in base) + "}"
    except Exception:
        return str(base)


def _dedup_key(code: str) -> str:
    """Trajectory code with numbers masked, for deduplication."""
    return re.sub(r"\b\d+\.\d+\b", "#", re.sub(r"\s+", " ", code)).strip()


def _md5(path: str) -> str:
    """Content hash of a file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    """Build tasks over all environments, excluding held-out tables by name and content."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-glob", action="append", required=True)
    ap.add_argument("--exclude-files", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-specs", type=int, default=0)
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = load_config(args.config)["frame"]
    held_names = {os.path.basename(p) for p in args.exclude_files}
    held_hashes = {_md5(p) for p in args.exclude_files if os.path.isfile(p)}
    specs = []
    for pattern in args.spec_glob:
        for p in sorted(glob.glob(pattern)):
            if os.path.basename(p) in held_names:
                continue
            if held_hashes and os.path.isfile(p) and _md5(p) in held_hashes:
                continue
            specs.append(p)
    rng.shuffle(specs)
    if args.max_specs:
        specs = specs[:args.max_specs]

    seen, n = set(), 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for i, spec in enumerate(specs):
            try:
                rels = load_relations(spec)
            except Exception:
                continue
            if not rels.tables or all(len(df) < 20 for df in rels.tables.values()):
                continue
            for t in tasks_for_relations(rels, rng, cfg, per_level=args.per_level):
                key = _dedup_key(t.code)
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps({
                    "task_type": "frame", "spec": spec, "gold_code": t.code, "level": t.level,
                    "depth": t.depth, "terminal": t.terminal, "schema_text": rels.schema_text(),
                    "target_repr": target_repr(t.base_output), "copy_seed": 0,
                }, ensure_ascii=False) + "\n")
                n += 1
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(specs)}] tasks={n}", flush=True)
    print(f"wrote {n} tasks -> {args.out}")


if __name__ == "__main__":
    main()
