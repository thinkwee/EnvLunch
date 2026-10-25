"""Perturbed database copies: delete rows of large tables and cascade along foreign keys."""

from __future__ import annotations

import os
import random
import shutil
import sqlite3
import threading
from typing import Optional

from envlunch.sql.db import load_schema, q

MIN_DROP_ROWS = 20


def _resolved_fks(schema) -> list:
    """Foreign keys with the referenced column resolved."""
    out = []
    for fk in schema.fks:
        if fk.from_col is None or fk.to_table not in schema.tables:
            continue
        to_col = fk.to_col
        if to_col is None:
            pks = schema.pks(fk.to_table)
            if len(pks) != 1:
                continue
            to_col = pks[0]
        out.append((fk.from_table, fk.from_col, fk.to_table, to_col))
    return out


def _dangling(ft, fc, tt, tc) -> str:
    """WHERE clause for rows whose parent was deleted in the copy."""
    return (f"{q(fc)} IS NOT NULL "
            f"AND {q(fc)} IN (SELECT {q(tc)} FROM base.{q(tt)} WHERE {q(tc)} IS NOT NULL) "
            f"AND {q(fc)} NOT IN (SELECT {q(tc)} FROM main.{q(tt)} WHERE {q(tc)} IS NOT NULL)")


def build_counterfactual(db_path: str, out_path: str, drop_frac: float = 0.3, seed: int = 0,
                         timeout: float = 30.0) -> Optional[dict]:
    """Write one perturbed copy; None if it is not referentially consistent."""
    rng = random.Random(seed)
    shutil.copyfile(db_path, out_path)
    fks = _resolved_fks(load_schema(db_path))
    ok = False
    try:
        conn = sqlite3.connect(out_path, timeout=10)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("ATTACH DATABASE ? AS base", (db_path,))
    except Exception:
        os.remove(out_path)
        return None
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM main.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [r[0] for r in cur.fetchall()]
        before, dropped = {}, 0
        for t in tables:
            cur.execute(f"SELECT rowid FROM main.{q(t)}")
            rids = [r[0] for r in cur.fetchall()]
            before[t] = len(rids)
            if len(rids) < MIN_DROP_ROWS:
                continue
            k = min(int(round(len(rids) * drop_frac)), len(rids) - 1)
            rng.shuffle(rids)
            for i in range(0, k, 500):
                chunk = rids[i:min(i + 500, k)]
                cur.execute(f"DELETE FROM main.{q(t)} WHERE rowid IN ({','.join('?' * len(chunk))})", chunk)
                dropped += max(cur.rowcount or 0, 0)
        for _ in range(50):
            changed = 0
            for ft, fc, tt, tc in fks:
                cur.execute(f"DELETE FROM main.{q(ft)} WHERE {_dangling(ft, fc, tt, tc)}")
                changed += max(cur.rowcount or 0, 0)
            if changed == 0:
                break
        conn.commit()
        left = 0
        for ft, fc, tt, tc in fks:
            cur.execute(f"SELECT COUNT(*) FROM main.{q(ft)} WHERE {_dangling(ft, fc, tt, tc)}")
            left += cur.fetchone()[0]
        emptied = False
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM main.{q(t)}")
            if cur.fetchone()[0] == 0 and before[t] > 0:
                emptied = True
        ok = dropped > 0 and left == 0 and not emptied
    except Exception:
        ok = False
    finally:
        timer.cancel()
        conn.close()
    if not ok:
        os.remove(out_path)
        return None
    return {"seed": seed, "drop_frac": drop_frac}


def build_copies(db_path: str, out_dir: str, k: int = 2, drop_frac: float = 0.3) -> list:
    """Build (or reuse) the K perturbed copies of a database."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(k):
        out = os.path.join(out_dir, f"copy_{i}.sqlite")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            paths.append(out)
            continue
        for seed in (1000 + i, 3000 + i, 5000 + i):
            if build_counterfactual(db_path, out, drop_frac, seed) is not None:
                paths.append(out)
                break
    return paths
