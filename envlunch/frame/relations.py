"""Loading dataframe environments and building their perturbed copies."""

from __future__ import annotations

import keyword
import os
import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_ROWS = int(os.environ.get("ENVLUNCH_MAX_ROWS", "50000"))
ID_NAME_RE = re.compile(r"(^id$|_id$|_code$|^code$|guid|uuid|^pk$|^index$|^unnamed)", re.IGNORECASE)


def measure_columns(df: pd.DataFrame, min_distinct: int = 11) -> list:
    """Numeric, non-identifier columns with enough distinct values."""
    return [c for c in df.columns
            if not ID_NAME_RE.search(str(c))
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique(dropna=True) >= min_distinct]


def category_columns(df: pd.DataFrame, max_card: int = 50) -> list:
    """Non-identifier columns with few distinct values."""
    return [c for c in df.columns
            if not ID_NAME_RE.search(str(c)) and 2 <= df[c].nunique(dropna=True) <= max_card]


@dataclass
class Relations:
    """Named dataframes and the foreign keys between them."""
    tables: dict
    fks: list = field(default_factory=list)

    @property
    def names(self) -> list:
        """Table names."""
        return list(self.tables)

    def schema_text(self) -> str:
        """Columns, dtypes, sizes and foreign keys as text."""
        lines = [f"  {n} ({len(df)} rows): " + ", ".join(f"{c}:{df[c].dtype}" for c in df.columns)
                 for n, df in self.tables.items()]
        if self.fks:
            lines.append("  FKs: " + "; ".join(f"{c}.{cc}->{p}.{pc}" for c, cc, p, pc in self.fks))
        return "\n".join(lines)


def _ident(name: str, used: set) -> str:
    """A unique Python identifier for a table name."""
    s = re.sub(r"\W", "_", str(name).strip())
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "t_" + s
    if keyword.iskeyword(s):
        s += "_"
    base, i = s, 1
    while s in used:
        s, i = f"{base}_{i}", i + 1
    used.add(s)
    return s


def _sanitize(tables: dict, fks: list) -> Relations:
    """Rename tables to valid identifiers."""
    used: set = set()
    ren = {n: _ident(n, used) for n in tables}
    return Relations({ren[n]: df for n, df in tables.items()},
                     [(ren.get(c, c), cc, ren.get(p, p), pc) for c, cc, p, pc in fks])


def _read(path: str) -> pd.DataFrame:
    """Read a CSV or Parquet file."""
    if path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path).head(MAX_ROWS)
    else:
        df = pd.read_csv(path, nrows=MAX_ROWS or None)
    df.columns = [str(c) for c in df.columns]
    return df


def load_relations(spec: str) -> Relations:
    """Load a CSV, Parquet, directory of tables, or SQLite database as Relations."""
    if os.path.isdir(spec):
        tables = {}
        for fn in sorted(os.listdir(spec)):
            if fn.endswith((".csv", ".parquet")):
                tables[os.path.splitext(fn)[0]] = _read(os.path.join(spec, fn))
        return _sanitize(tables, [])
    if spec.endswith((".sqlite", ".db", ".sqlite3")):
        con = sqlite3.connect(spec)
        try:
            names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            tables, fks = {}, []
            for n in names:
                df = pd.read_sql_query(f'SELECT * FROM "{n}" LIMIT {MAX_ROWS}', con)
                df.columns = [str(c) for c in df.columns]
                tables[n] = df
                for r in con.execute(f'PRAGMA foreign_key_list("{n}")'):
                    fks.append((n, str(r[3]), str(r[2]), str(r[4] or "")))
            return _sanitize(tables, fks)
        finally:
            con.close()
    return Relations({"df": _read(spec)})


def perturb(rels: Relations, seed: int, frac: float = 0.3, amp: float = 0.4) -> dict:
    """One copy: rescale measure columns, delete rows of large tables, cascade along foreign keys."""
    rng = np.random.RandomState(seed)
    fk_cols = {(c, cc) for c, cc, _, _ in rels.fks}
    out = {}
    for n, df in rels.tables.items():
        d = df.copy().reset_index(drop=True)
        for c in measure_columns(d):
            if (n, c) in fk_cols:
                continue
            f = rng.uniform(1 - amp, 1 + amp)
            if abs(f - 1.0) < 0.1:
                f = 1.0 + (0.1 if f >= 1.0 else -0.1)
            d[c] = d[c] * f
        out[n] = d
    for n, d in out.items():
        if len(d) >= 20:
            k = max(1, int(round(len(d) * frac)))
            out[n] = d.iloc[sorted(rng.permutation(len(d))[k:])].reset_index(drop=True)
    for _ in range(max(1, len(rels.fks))):
        changed = False
        for child, ccol, parent, pcol in rels.fks:
            if child not in out or parent not in out or ccol not in out[child].columns:
                continue
            keys = set(out[parent][pcol]) if pcol in out[parent].columns else set(out[parent].iloc[:, 0])
            before = len(out[child])
            out[child] = out[child][out[child][ccol].isin(keys)].reset_index(drop=True)
            changed |= len(out[child]) != before
        if not changed:
            break
    return out


def perturbed_copies(rels: Relations, k: int = 3, seed: int = 0, frac: float = 0.3, amp: float = 0.4) -> list:
    """K perturbed copies that keep every table non-empty."""
    copies = []
    for i in range(k):
        w = perturb(rels, seed + 1 + i, frac, amp)
        if all(len(df) > 0 for df in w.values()):
            copies.append(tuple(w.items()))
    return copies
