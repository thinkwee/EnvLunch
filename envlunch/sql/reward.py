"""Agreement of SQL results as overlap of row multisets, minimum over copies."""

from __future__ import annotations

from collections import Counter
from typing import Optional

from envlunch.sql.db import execute


def _rows(db_path: str, sql: str, timeout: float = 8.0) -> Optional[list]:
    """Rows of a query, or None on error or truncation."""
    r = execute(db_path, sql, timeout=timeout, max_rows=2000)
    if not r["ok"] or r["truncated"]:
        return None
    return r["rows"]


def norm_cell(v) -> str:
    """Canonical string of a cell."""
    if v is None:
        return "\x00NULL"
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:.6g}"
    return str(v).strip()


def _multiset(rows: list) -> Counter:
    """Result rows as a multiset."""
    return Counter(tuple(norm_cell(c) for c in row) for row in rows)


def jaccard(a: Counter, b: Counter) -> float:
    """Multiset overlap of two results."""
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    union = sum(max(a[k], b[k]) for k in keys)
    return sum(min(a[k], b[k]) for k in keys) / union if union else 0.0


def agreement(agent_rows: Optional[list], gold_rows: Optional[list]) -> float:
    """m for one copy: 1 when the multisets are equal, otherwise based on their overlap."""
    if agent_rows is None or gold_rows is None:
        return 0.0
    a, g = _multiset(agent_rows), _multiset(gold_rows)
    return 1.0 if a == g else 0.5 * jaccard(a, g)


def score(agent_sql: Optional[str], gold_sql: str, copies: list) -> float:
    """Minimum agreement of the agent's query with the hidden query over the copies."""
    if not agent_sql or not copies:
        return 0.0
    return min(agreement(_rows(p, agent_sql), _rows(p, gold_sql)) for p in copies)

