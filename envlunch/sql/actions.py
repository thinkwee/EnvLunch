"""The fifteen SQL operators, with parameters instantiated from the database."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

from envlunch.sql.db import Schema, execute, q

ID_NAME_RE = re.compile(r"(^id$|_id$|_code$|^code$|guid|uuid|^pk$)", re.IGNORECASE)
DATE_NAME_RE = re.compile(r"(date|time|year|month|day|_at$|_on$)", re.IGNORECASE)
AGGS = ("SUM", "AVG", "COUNT", "MIN", "MAX")
AGG_LEVELS = tuple(range(1, 11))
PLAIN_FAMILIES = ("value", "null", "rank", "time", "joinproj")


@dataclass
class ColInfo:
    """A column and its role (measure, category, entity or excluded)."""
    name: str
    cls: str
    distinct: int


def _rows(db_path: str, sql: str, timeout: float = 5.0) -> Optional[list]:
    """Rows of a query, or None on error."""
    r = execute(db_path, sql, timeout=timeout)
    return r["rows"] if r["ok"] else None


def classify_columns(schema: Schema, table: str, db_path: str) -> dict:
    """Assign each column of a table a role from its type and distinct values."""
    pks = set(schema.pks(table))
    fk_cols = {fk.from_col for fk in schema.fks if fk.from_table == table}
    out = {}
    for c in schema.cols(table):
        r = _rows(db_path, f"SELECT COUNT(DISTINCT {q(c.name)}), COUNT(*) FROM {q(table)}")
        distinct, total = (r[0] if r else (0, 0))
        if c.name in pks:
            cls = "entity"
        elif distinct <= 1 or total == 0 or c.name in fk_cols or ID_NAME_RE.search(c.name):
            cls = "excluded"
        elif c.is_numeric and distinct > 10:
            cls = "measure"
        elif 2 <= distinct <= 50:
            cls = "category"
        else:
            cls = "excluded"
        out[c.name] = ColInfo(c.name, cls, distinct)
    return out


def _literal(db_path: str, table: str, col: str, rng: random.Random) -> Optional[str]:
    """A value that the column actually holds, as an SQL literal."""
    r = _rows(db_path, f"SELECT DISTINCT {q(col)} FROM {q(table)} WHERE {q(col)} IS NOT NULL LIMIT 50")
    if not r:
        return None
    v = rng.choice(r)[0]
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, (int, float)):
        return str(v)
    return None


def _pick(cls_map: dict, kind: str, rng: random.Random, lo: int = 0, hi: int = 10 ** 9) -> Optional[str]:
    """A random column of the given role within a distinct-value range."""
    c = [x.name for x in cls_map.values() if x.cls == kind and lo <= x.distinct <= hi]
    return rng.choice(c) if c else None


def _group_key(cls_map: dict, rng: random.Random) -> Optional[str]:
    """A category column suitable for grouping."""
    return _pick(cls_map, "category", rng, 2, 30) or _pick(cls_map, "category", rng, 0, 30)


def _fk_neighbors(schema: Schema, table: str, exclude: str = None) -> list:
    """Foreign-key neighbours of a table that exist in the schema."""
    return [(o, m, c) for (o, m, c) in schema.neighbors(table)
            if o in schema.tables and m and c and o != exclude]


def _order(cols: list) -> str:
    """ORDER BY list for the given columns."""
    return ", ".join(c if c.startswith(("a.", "b.")) else q(c) for c in cols)


def _date_col(db_path: str, table: str, cols: list) -> Optional[str]:
    """A column that parses as a year, if any."""
    for c in cols:
        if not DATE_NAME_RE.search(c):
            continue
        r = _rows(db_path, f"SELECT strftime('%Y', {q(c)}) FROM {q(table)} WHERE {q(c)} IS NOT NULL LIMIT 200")
        if not r:
            continue
        vals = [x[0] for x in r]
        good = [v for v in vals if v and v.lstrip("-").isdigit() and 1000 <= int(v) <= 2100]
        if len(good) >= 0.8 * len(vals) and len(set(good)) >= 2:
            return c
    return None


def _null_cols(db_path: str, table: str, cols: list, rng: random.Random) -> list:
    """Columns with some but not all values NULL."""
    out = []
    for c in cols[:8]:
        r = _rows(db_path, f"SELECT SUM(CASE WHEN {q(c)} IS NULL THEN 1 ELSE 0 END), COUNT(*) FROM {q(table)}")
        if r and r[0][0] and r[0][1] and 0 < r[0][0] < r[0][1]:
            out.append(c)
    rng.shuffle(out)
    return out


def sample_aggregate(schema: Schema, db_path: str, rng: random.Random, level: int,
                     allowed: Optional[set] = None) -> Optional[dict]:
    """Instantiate one of the ten aggregating operators (level 1-10) on a valid table."""
    tables = [t for t in schema.tables if allowed is None or t in allowed]
    rng.shuffle(tables)
    for table in tables:
        cls_map = classify_columns(schema, table, db_path)
        cat = _group_key(cls_map, rng)
        if cat is None:
            continue
        meas = _pick(cls_map, "measure", rng)
        agg = rng.choice([a for a in AGGS if a != "COUNT"]) if meas and rng.random() < 0.8 else "COUNT"
        if level in (5, 7, 8, 9) and meas is None:
            continue
        agg_expr = "COUNT(*)" if agg == "COUNT" else f"{agg}({q(meas)})"
        alias = ("n" if agg == "COUNT" else f"{agg.lower()}_{meas}")[:30]

        if level == 1:
            sql = f"SELECT {q(cat)}, {agg_expr} AS {q(alias)}\nFROM {q(table)}\nGROUP BY {q(cat)}"
        elif level == 2:
            cands = [c.name for c in cls_map.values() if c.cls in ("category", "measure") and c.name != cat]
            if not cands:
                continue
            fcol = rng.choice(cands)
            val = _literal(db_path, table, fcol, rng)
            if val is None:
                continue
            op = "=" if cls_map[fcol].cls == "category" else rng.choice([">", "<", ">="])
            sql = (f"SELECT {q(cat)}, {agg_expr} AS {q(alias)}\nFROM {q(table)}\n"
                   f"WHERE {q(fcol)} {op} {val}\nGROUP BY {q(cat)}")
        elif level in (3, 4):
            nbrs = _fk_neighbors(schema, table)
            if not nbrs:
                continue
            ot, mc, oc = rng.choice(nbrs)
            ocat = _group_key(classify_columns(schema, ot, db_path), rng)
            if ocat is None:
                continue
            join = f"FROM {q(table)} AS a\nJOIN {q(ot)} AS b ON a.{q(mc)} = b.{q(oc)}"
            key = f"b.{q(ocat)}"
            agg_b = "COUNT(*)" if agg == "COUNT" else f"{agg}(a.{q(meas)})"
            sql = f"SELECT {key} AS {q(ocat)}, {agg_b} AS {q(alias)}\n{join}\nGROUP BY {key}"
            if level == 4:
                sql += (f"\nHAVING {agg_b} > (SELECT AVG(x) FROM "
                        f"(SELECT {agg_b} AS x {join} GROUP BY {key}))")
        elif level == 5:
            sql = (f"SELECT {q(cat)}, {agg_expr} AS {q(alias)}\nFROM {q(table)}\n"
                   f"WHERE {q(meas)} > (SELECT AVG({q(meas)}) FROM {q(table)})\nGROUP BY {q(cat)}")
        elif level == 6:
            cands = [c.name for c in cls_map.values() if c.cls == "category" and 2 <= c.distinct <= 12]
            if len(cands) < 2:
                continue
            c1, c2 = rng.sample(cands, 2)
            sql = (f"SELECT {q(c1)}, {q(c2)}, {agg_expr} AS {q(alias)}\nFROM {q(table)}\n"
                   f"GROUP BY {q(c1)}, {q(c2)}")
        elif level == 7:
            r = _rows(db_path, f"SELECT AVG({q(meas)}) FROM {q(table)}")
            if not r or r[0][0] is None:
                continue
            sql = (f"SELECT {q(cat)}, SUM(CASE WHEN {q(meas)} > {float(r[0][0])} THEN 1 ELSE 0 END) AS n_high, "
                   f"{agg_expr} AS {q(alias)}\nFROM {q(table)}\nGROUP BY {q(cat)}")
        elif level == 8:
            ent = _pick(cls_map, "entity", rng)
            if ent is None or ent == cat:
                continue
            sql = (f"SELECT {q(cat)}, {q(ent)}, {q(meas)}\nFROM (\n"
                   f"  SELECT {q(cat)}, {q(ent)}, {q(meas)}, ROW_NUMBER() OVER (PARTITION BY {q(cat)} "
                   f"ORDER BY {q(meas)} DESC, {q(ent)} ASC) AS rnk\n  FROM {q(table)}\n) AS w\nWHERE rnk <= 2")
        elif level == 9:
            nb1 = _fk_neighbors(schema, table)
            if not nb1:
                continue
            ot, mc, oc = rng.choice(nb1)
            nb2 = _fk_neighbors(schema, ot, exclude=table)
            if not nb2:
                continue
            o2, m2, c2 = rng.choice(nb2)
            ocat = _group_key(classify_columns(schema, o2, db_path), rng)
            if ocat is None:
                continue
            agg_a = "COUNT(*)" if agg == "COUNT" else f"{agg}(a.{q(meas)})"
            sql = (f"SELECT c.{q(ocat)} AS {q(ocat)}, {agg_a} AS {q(alias)}\nFROM {q(table)} AS a\n"
                   f"JOIN {q(ot)} AS b ON a.{q(mc)} = b.{q(oc)}\nJOIN {q(o2)} AS c ON b.{q(m2)} = c.{q(c2)}\n"
                   f"GROUP BY c.{q(ocat)}")
        elif level == 10:
            sql = (f"WITH grp AS (\n  SELECT {q(cat)} AS k, {agg_expr} AS m\n  FROM {q(table)}\n"
                   f"  GROUP BY {q(cat)}\n)\nSELECT k, m FROM grp WHERE m > (SELECT AVG(m) FROM grp)")
        else:
            return None
        return {"sql": sql, "table": table, "level": level, "family": "agg"}
    return None


def sample_plain(schema: Schema, db_path: str, rng: random.Random, family: str,
                 allowed: Optional[set] = None) -> Optional[dict]:
    """Instantiate one of the five non-aggregating operators on a valid table."""
    tables = [t for t in schema.tables if allowed is None or t in allowed]
    rng.shuffle(tables)
    level = 11 + PLAIN_FAMILIES.index(family)
    for table in tables:
        cls_map = classify_columns(schema, table, db_path)
        names = [c.name for c in schema.cols(table)]
        cat = _group_key(cls_map, rng)
        meas = _pick(cls_map, "measure", rng)
        ent = _pick(cls_map, "entity", rng)
        sql = None

        if family == "value" and cat and ent:
            lit = _literal(db_path, table, cat, rng)
            if lit is not None and len(lit) <= 42:
                proj = [c for c in (ent, meas) if c]
                sql = (f"SELECT DISTINCT {', '.join(q(c) for c in proj)}\nFROM {q(table)}\n"
                       f"WHERE {q(cat)} = {lit}\nORDER BY {_order(proj)}\nLIMIT 20")
        elif family == "null" and ent:
            nc = _null_cols(db_path, table, names, rng)
            if nc:
                sql = (f"SELECT DISTINCT {q(ent)}, {q(nc[0])}\nFROM {q(table)}\nWHERE {q(nc[0])} IS NULL\n"
                       f"ORDER BY {_order([ent, nc[0]])}\nLIMIT 20")
        elif family == "rank" and meas and ent:
            k = rng.choice([3, 5, 10])
            r = _rows(db_path, f"SELECT {q(meas)} FROM {q(table)} WHERE {q(meas)} IS NOT NULL "
                               f"ORDER BY {q(meas)} DESC LIMIT {k + 1}")
            if r and len(r) >= k + 1 and r[k - 1][0] != r[k][0]:
                sql = (f"SELECT {q(ent)}, {q(meas)}\nFROM {q(table)}\nWHERE {q(meas)} IS NOT NULL\n"
                       f"ORDER BY {q(meas)} DESC, {q(ent)} ASC\nLIMIT {k}")
        elif family == "time":
            dc = _date_col(db_path, table, names)
            if dc:
                agg = "COUNT(*)" if meas is None else f"SUM({q(meas)})"
                alias = "n" if meas is None else f"sum_{meas}"[:30]
                yr = f"strftime('%Y', {q(dc)})"
                sql = (f"SELECT {yr} AS yr, {agg} AS {q(alias)}\nFROM {q(table)}\n"
                       f"WHERE {yr} IS NOT NULL\nGROUP BY {yr}\nORDER BY yr")
        elif family == "joinproj" and ent and cat:
            nbrs = _fk_neighbors(schema, table)
            rng.shuffle(nbrs)
            for ot, mc, oc in nbrs:
                if allowed is not None and ot not in allowed:
                    continue
                ocat = _group_key(classify_columns(schema, ot, db_path), rng)
                if ocat is None:
                    continue
                lit = _literal(db_path, ot, ocat, rng)
                if lit is None or len(lit) > 42:
                    continue
                proj = [f"a.{q(ent)}", f"a.{q(cat)}"]
                sql = (f"SELECT DISTINCT {', '.join(proj)}\nFROM {q(table)} AS a\n"
                       f"JOIN {q(ot)} AS b ON a.{q(mc)} = b.{q(oc)}\nWHERE b.{q(ocat)} = {lit}\n"
                       f"ORDER BY {', '.join(proj)}\nLIMIT 20")
                break
        if sql:
            return {"sql": sql, "table": table, "level": level, "family": family}
    return None
