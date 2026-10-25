"""SQL trajectory synthesis and filtering -> tasks.jsonl."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter
from pathlib import Path

from envlunch.env import load_config
from envlunch.sql import actions as A
from envlunch.sql import reward as R
from envlunch.sql.db import execute, load_schema, schema_text, table_sizes
from envlunch.sql.perturb import build_copies

MAX_TABLE_ROWS = 30000
MAX_DB_FILE_MB = 800
MIN_ROWS, MAX_ROWS = 1, 30
OPERATORS = [("agg", lvl) for lvl in A.AGG_LEVELS] + [(f, None) for f in A.PLAIN_FAMILIES]
GIVE_UP = 40


def non_trivial(cols: list, rows: list, family: str) -> bool:
    """Whether a result is non-degenerate (non-empty, not constant)."""
    if not rows:
        return False
    if family == "agg":
        vals = [r[-1] for r in rows]
        if all(v is None for v in vals):
            return False
        return not (len(vals) > 1 and len({str(v) for v in vals}) == 1)
    if len(rows) == 1:
        return True
    return any(len({str(r[i]) for r in rows}) > 1 for i in range(len(cols)))


def tasks_for_db(db_path: str, copy_dir: str, rng: random.Random, per_db: int, cfg: dict) -> tuple:
    """Synthesise tasks on one database: draw valid operators uniformly, keep a task only if it is non-degenerate in every copy and changes in at least one."""
    stats = Counter()
    if os.path.getsize(db_path) > MAX_DB_FILE_MB * 1024 * 1024:
        return [], stats
    schema = load_schema(db_path)
    allowed = {t for t, n in table_sizes(db_path).items() if 0 <= n <= MAX_TABLE_ROWS}
    if not schema.tables or not allowed:
        return [], stats
    copies = build_copies(db_path, copy_dir, cfg["copies"], cfg["drop_frac"])
    if len(copies) < cfg["copies"]:
        stats["copy_failed"] += 1
        return [], stats

    stext = schema_text(db_path)
    tasks, seen = [], set()
    fail = {op: 0 for op in OPERATORS}
    for _ in range(per_db * 20):
        if len(tasks) >= per_db:
            break
        live = [op for op in OPERATORS if fail[op] < GIVE_UP]
        if not live:
            break
        op = rng.choice(live)
        fam, level = op
        fail[op] += 1
        if fam == "agg":
            spec = A.sample_aggregate(schema, db_path, rng, level, allowed)
        else:
            spec = A.sample_plain(schema, db_path, rng, fam, allowed)
        if spec is None or spec["sql"] in seen:
            stats["no_action"] += 1
            continue
        sql = spec["sql"]
        if "LIMIT" in sql.upper() and "ORDER BY" not in sql.upper():
            continue
        base = execute(db_path, sql, max_rows=40)
        if not base["ok"] or not (MIN_ROWS <= len(base["rows"]) <= MAX_ROWS):
            stats["bad_rows"] += 1
            continue
        if not non_trivial(base["cols"], base["rows"], fam):
            stats["trivial"] += 1
            continue
        outs = [execute(p, sql, max_rows=2000) for p in copies]
        if not all(r["ok"] and not r["truncated"] and non_trivial(r["cols"], r["rows"], fam) for r in outs):
            stats["degenerate_in_copy"] += 1
            continue
        if all(R.agreement(r["rows"], base["rows"]) == 1.0 for r in outs):
            stats["unchanged_in_copies"] += 1
            continue
        seen.add(sql)
        fail[op] = 0
        tasks.append({
            "task_type": "sql", "db_path": db_path, "db_name": Path(db_path).stem,
            "schema_text": stext, "gold_sql": sql, "level": spec["level"], "family": fam,
            "header_mode": "hard" if rng.random() < 0.3 else "easy",
            "result_cols": list(base["cols"]), "result_rows": [list(r) for r in base["rows"]],
            "copies": copies,
        })
        stats[f"ok_{fam}"] += 1
    return tasks, stats


def main():
    """Build tasks over all databases and write them as jsonl."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-glob", action="append", required=True)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--copy-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-db", type=int, default=120)
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    excluded = set()
    if args.exclude:
        excluded = {l.strip() for l in open(args.exclude) if l.strip()}
    dbs, names = [], set()
    for pattern in args.db_glob:
        for p in sorted(glob.glob(pattern)):
            name = Path(p).stem
            if name in excluded or name in names:
                continue
            names.add(name)
            dbs.append(p)
    rng = random.Random(args.seed)
    cfg = load_config(args.config)["sql"]
    rng.shuffle(dbs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    total, n = Counter(), 0
    with open(args.out, "w") as f:
        for i, p in enumerate(dbs):
            if n >= args.target:
                break
            tasks, stats = tasks_for_db(p, os.path.join(args.copy_dir, Path(p).stem), rng, args.per_db, cfg)
            total.update(stats)
            for t in tasks:
                f.write(json.dumps(t) + "\n")
            n += len(tasks)
            print(f"[{i + 1}/{len(dbs)}] {Path(p).stem}: +{len(tasks)} (total {n})", flush=True)
    print(json.dumps(dict(total), indent=2))


if __name__ == "__main__":
    main()
