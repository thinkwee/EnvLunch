"""Dataframe action chains (source, transforms, terminal), synthesised against the live frame."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from envlunch.frame.artifact import normalize
from envlunch.frame.relations import Relations, category_columns, measure_columns, perturbed_copies
from envlunch.frame.sandbox import SAFE_BUILTINS

AGG_FNS = ["mean", "sum", "max", "min", "std", "median"]
LEVELS = {"easy": (0,), "medium": (1, 2), "hard": (3, 4), "expert": (5, 6)}
MIN_ROWS_FOR_FILTER = 24
MIN_ROWS_FOR_COUNT = 5
MIN_COUNT_ANSWER = 3.0


@dataclass
class Task:
    """A synthesised trajectory as code, with its level and outputs."""
    code: str
    level: str
    depth: int
    terminal: str
    base_output: object = None
    copy_outputs: list = field(default_factory=list)


def _ns(frames: dict) -> dict:
    """Execution namespace with the tables."""
    return {"__builtins__": dict(SAFE_BUILTINS), "pd": pd, "np": np, **frames}


def run_gold(frames: dict, code: str):
    """Run a trajectory on the given tables and return its normalised result."""
    try:
        g = _ns({n: f.copy() for n, f in frames.items()})
        exec(code, g)
        return normalize(g.get("result"))
    except Exception:
        return None


def _pick(seq, rng):
    """Random element or None."""
    return rng.choice(seq) if seq else None


def _relaxed_measures(df) -> list:
    """Measure columns, falling back to a lower distinct-value threshold."""
    return measure_columns(df) or measure_columns(df, min_distinct=3)


def _pipeline(rels: Relations, rng: random.Random, n_ops: int, use_join: bool):
    """Source action followed by n transforms, each drawn from those valid on the current frame."""
    g = _ns(dict(rels.tables))
    lines = []

    def run(line):
        """Append one action and execute it."""
        lines.append(line)
        exec(line, g)

    if use_join and rels.fks:
        child, ccol, parent, pcol = _pick(rels.fks, rng)
        if not pcol:
            return None
        run(f"r = {child}.merge({parent}, left_on={ccol!r}, right_on={pcol!r}, how='inner', suffixes=('', '_p'))")
    else:
        run(f"r = {_pick(rels.names, rng)}.copy()")
    if len(g["r"]) == 0:
        return None

    for _ in range(n_ops):
        cur = g["r"]
        meas, cats = measure_columns(cur), category_columns(cur)
        roomy = len(cur) >= MIN_ROWS_FOR_FILTER
        ops = []
        if meas and roomy:
            ops += ["filter_threshold", "topk"]
        if cats and roomy:
            ops += ["filter_category"]
        if meas and cats:
            ops += (["filter_both"] if roomy else []) + ["groupby"]
        if len(cats) >= 2 and meas:
            ops += ["groupby_two"]
        if len(meas) >= 2:
            ops += ["ratio"]
        if meas:
            ops += ["rank"]
        op = _pick(ops, rng)
        if op is None:
            break
        if op == "filter_threshold":
            m = _pick(meas, rng)
            run(f"r = r[r[{m!r}] > {float(cur[m].quantile(rng.choice([0.4, 0.5, 0.6])))!r}]")
        elif op == "filter_category":
            c = _pick(cats, rng)
            run(f"r = r[r[{c!r}] == {_pick(list(cur[c].value_counts().index[:10]), rng)!r}]")
        elif op == "filter_both":
            c = _pick(cats, rng)
            m = _pick([x for x in meas if x != c], rng)
            if m is None:
                continue
            v = _pick(list(cur[c].value_counts().index[:10]), rng)
            run(f"r = r[(r[{c!r}] == {v!r}) & (r[{m!r}] > {float(cur[m].quantile(0.5))!r})]")
        elif op == "topk":
            run(f"r = r.sort_values({_pick(meas, rng)!r}, ascending=False).head({rng.choice([3, 5, 10])})")
        elif op == "groupby":
            c = _pick(cats, rng)
            m = _pick([x for x in meas if x != c], rng)
            if m is None:
                continue
            run(f"r = r.groupby({c!r})[{m!r}].{_pick(AGG_FNS, rng)}().reset_index()")
        elif op == "groupby_two":
            c1, c2 = rng.sample(cats, 2)
            m = _pick([x for x in meas if x not in (c1, c2)], rng)
            if m is None:
                continue
            run(f"r = r.groupby([{c1!r}, {c2!r}])[{m!r}].{_pick(AGG_FNS, rng)}().reset_index()")
        elif op == "ratio":
            m1, m2 = rng.sample(meas, 2)
            name = f"{m1}_over_{m2}"[:40]
            run(f"r = r.assign(**{{{name!r}: r[{m1!r}] / r[{m2!r}].replace(0, np.nan)}})")
        elif op == "rank":
            run(f"r = r.assign(_rank=r[{_pick(meas, rng)!r}].rank(method='min', ascending=False))")
        if len(g["r"]) == 0:
            return None
    return lines, g["r"]


def sample_task(rels: Relations, rng: random.Random, level: str):
    """One trajectory at a difficulty level, ending in a terminal action."""
    n_ops = rng.choice(LEVELS[level])
    use_join = level in ("hard", "expert") and bool(rels.fks) and rng.random() < 0.5
    built = _pipeline(rels, rng, n_ops, use_join)
    if built is None:
        return None
    lines, cur = built
    meas, cats = _relaxed_measures(cur), category_columns(cur)
    terminal = ""
    if cats and meas and rng.random() < 0.3:
        c = _pick(cats, rng)
        m = _pick([x for x in meas if x != c], rng)
        if m is not None and cur[c].nunique() <= 50:
            terminal = "mapping"
            lines.append(f"result = {{str(k): float(v) for k, v in r.groupby({c!r})[{m!r}].{_pick(AGG_FNS, rng)}().items()}}")
    if not terminal:
        kinds = (["scalar"] * 4 + ["quantile"] if meas else []) + (["nunique"] if cats else [])
        if len(cur) >= MIN_ROWS_FOR_COUNT:
            kinds.append("count")
        terminal = _pick(kinds, rng)
        if terminal is None:
            return None
        if terminal == "count":
            lines.append("result = float(len(r))")
        elif terminal == "nunique":
            lines.append(f"result = float(r[{_pick(cats, rng)!r}].nunique())")
        elif terminal == "quantile":
            lines.append(f"result = float(r[{_pick(meas, rng)!r}].quantile({rng.choice([0.75, 0.9, 0.95])!r}))")
        else:
            lines.append(f"result = float(r[{_pick(meas, rng)!r}].{_pick(AGG_FNS, rng)}())")
    return Task("\n".join(lines), level, n_ops + int(use_join), terminal)


def _same(a, b) -> bool:
    """Whether two results are equal up to tolerance."""
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(ka == kb and _same(va, vb) for (ka, va), (kb, vb) in zip(a, b))
    return a == b


def validate(rels: Relations, task: Task, cfg: dict) -> bool:
    """Keep a trajectory only if it is non-degenerate in every copy and changes in at least one."""
    base = run_gold(rels.tables, task.code)
    if base is None or base == 0.0:
        return False
    if task.terminal in ("count", "nunique") and isinstance(base, float) and base < MIN_COUNT_ANSWER:
        return False
    copies = perturbed_copies(rels, cfg["copies"], 0, cfg["drop_frac"], cfg["scale_amp"])
    if len(copies) < cfg["copies"]:
        return False
    outs = [run_gold(dict(w), task.code) for w in copies]
    task.base_output, task.copy_outputs = base, outs
    if any(o is None or o == 0.0 for o in outs):
        return False
    return any(not _same(o, base) for o in outs)


def tasks_for_relations(rels: Relations, rng: random.Random, cfg: dict, per_level: int = 6, tries: int = 12) -> list:
    """Synthesise up to per_level tasks for each difficulty level."""
    kept = []
    quota = max(1, per_level // 3) if per_level >= 3 else per_level
    for level in LEVELS:
        got, by_terminal = 0, {}
        for _ in range(per_level * tries):
            if got >= per_level:
                break
            try:
                t = sample_task(rels, rng, level)
                ok = t is not None and validate(rels, t, cfg)
            except Exception:
                ok = False
            if not ok or by_terminal.get(t.terminal, 0) >= quota:
                continue
            by_terminal[t.terminal] = by_terminal.get(t.terminal, 0) + 1
            kept.append(t)
            got += 1
    return kept
