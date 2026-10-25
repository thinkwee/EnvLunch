"""SQLite schema reading, read-only execution and observation formatting."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional

NUMERIC_TYPES = {
    "INTEGER", "INT", "TINYINT", "SMALLINT", "MEDIUMINT", "BIGINT",
    "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL",
}


@dataclass
class Column:
    """A table column."""
    name: str
    type: str
    is_pk: bool

    @property
    def is_numeric(self) -> bool:
        """Whether the declared type is numeric."""
        return self.type.upper().split("(")[0].strip() in NUMERIC_TYPES


@dataclass
class FK:
    """A foreign key."""
    from_table: str
    from_col: str
    to_table: str
    to_col: Optional[str]


@dataclass
class Schema:
    """Tables, columns and foreign keys of a database."""
    db_path: str
    tables: dict = field(default_factory=dict)
    fks: list = field(default_factory=list)

    def cols(self, table: str) -> list:
        """Columns of a table."""
        return self.tables.get(table, [])

    def pks(self, table: str) -> list:
        """Primary-key column names of a table."""
        return [c.name for c in self.cols(table) if c.is_pk]

    def neighbors(self, table: str) -> list:
        """Tables joined to this one by a foreign key, with the join columns."""
        out = []
        for fk in self.fks:
            if fk.from_table == table:
                out.append((fk.to_table, fk.from_col, fk.to_col))
            elif fk.to_table == table:
                out.append((fk.from_table, fk.to_col, fk.from_col))
        return out


def q(name: str) -> str:
    """Quote an SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def load_schema(db_path: str) -> Schema:
    """Read the schema of a database."""
    schema = Schema(db_path=db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    names = [r[0] for r in cur.fetchall()]
    for t in names:
        cur.execute(f"PRAGMA table_info({q(t)})")
        schema.tables[t] = [Column(r[1], r[2] or "TEXT", r[5] > 0) for r in cur.fetchall()]
    for t in names:
        cur.execute(f"PRAGMA foreign_key_list({q(t)})")
        for r in cur.fetchall():
            if r[2] in schema.tables:
                schema.fks.append(FK(t, r[3], r[2], r[4]))
    conn.close()
    return schema


def table_sizes(db_path: str, timeout: float = 5.0) -> dict:
    """Row count of every table."""
    sizes = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        names = [r[0] for r in cur.fetchall()]
    except Exception:
        return sizes
    for t in names:
        timer = threading.Timer(timeout, conn.interrupt)
        timer.start()
        try:
            cur.execute(f"SELECT COUNT(*) FROM {q(t)}")
            sizes[t] = cur.fetchone()[0]
        except Exception:
            sizes[t] = -1
        finally:
            timer.cancel()
    conn.close()
    return sizes


def schema_text(db_path: str) -> str:
    """CREATE statements of all tables."""
    out = execute(db_path, "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    if not out["ok"]:
        return ""
    return "\n".join(r[0] for r in out["rows"] if r[0])


def execute(db_path: str, sql: str, timeout: float = 8.0, max_rows: int = 1000) -> dict:
    """Run a read-only query with a timeout; return columns and rows or an error."""
    if not sql or not sql.strip():
        return {"ok": False, "err": "empty query"}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        conn.execute("PRAGMA query_only = ON")
    except Exception as e:
        return {"ok": False, "err": str(e)[:200]}
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows + 1)
    except Exception as e:
        return {"ok": False, "err": f"sql_err: {str(e)[:200]}"}
    finally:
        timer.cancel()
        try:
            conn.close()
        except Exception:
            pass
    return {"ok": True, "cols": cols, "rows": [tuple(r) for r in rows[:max_rows]],
            "truncated": len(rows) > max_rows}


def format_obs(obs: dict, max_rows: int = 10, max_row_chars: int = 120, max_chars: int = 600) -> str:
    """Render a query result as a truncated observation."""
    if not obs.get("ok"):
        return f"<obs err='{obs.get('err', '')[:200]}'/>"
    rows = obs.get("rows", [])
    lines = [f"cols={obs.get('cols', [])}"]
    for r in rows[:max_rows]:
        line = str(list(r))
        lines.append(line[:max_row_chars] + "..." if len(line) > max_row_chars else line)
    if obs.get("truncated"):
        lines.append(f"... (truncated, only first {len(rows)} shown)")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return f"<obs>\n{text}\n</obs>"
