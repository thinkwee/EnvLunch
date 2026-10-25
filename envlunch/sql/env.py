"""Relational warm-up environment."""

from __future__ import annotations

from envlunch.env import Env
from envlunch.sql.db import execute, format_obs
from envlunch.sql.reward import norm_cell, score

SYSTEM = """You are an expert SQL agent working on a SQLite database.

Each turn, emit EXACTLY ONE of:

  <sql>your SQL query</sql>
    Run a read-only query to inspect the data. System returns at most 20 rows.

  <answer>your final answer</answer>
    Your final answer. For most tasks this is ONE complete SELECT statement.
    For "find the value" tasks it is the single value requested.

SQL tips for SQLite:
  - Reserved keywords as identifiers MUST be double-quoted, e.g. FROM "transaction".
  - Aggregates: SUM/AVG/COUNT/MIN/MAX. JOINs follow foreign keys.

Strategy:
  1. Use <sql> to explore tables, columns, and values.
  2. Test hypotheses by running candidate queries and comparing to the target.
  3. By turn {max_turns}, you MUST <answer>; unanswered tasks score 0.

Do not chat. Emit only <sql>...</sql> or <answer>...</answer>."""

SAME_ERROR_HINT = ('\n HINT: same error as last turn. Reserved-word identifiers need double quotes, '
                   'e.g. FROM "transaction".')


def render_table(cols: list, rows: list, header_mode: str = "easy") -> str:
    """Target result as a markdown table."""
    headers = [f"col_{i}" for i in range(len(cols))] if header_mode == "hard" else [str(c) for c in cols]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(norm_cell(c).replace("|", "\\|") for c in row) + " |" for row in rows]
    return "\n".join(lines)


class SqlEnv(Env):
    """Recover an SQL query from its result table by querying the database."""
    name = "sql"
    tag = "sql"
    system = SYSTEM

    def __init__(self, task, config=None):
        """Bind the task and remember the last error."""
        super().__init__(task, config)
        self.last_err = None

    def user_message(self) -> str:
        """Target result table and schema."""
        t = self.task
        table = render_table(t["result_cols"], t["result_rows"], t.get("header_mode", "easy"))
        return (
            "## Target result table\n"
            "The following is the COMPLETE output of a hidden SQL query on this database:\n\n"
            f"{table}\n\n"
            "## Your job\n"
            "Find a SQL query that produces EXACTLY this result (same rows, order ignored). "
            "Your query will be tested on a DIFFERENT snapshot of this database, so it must capture the "
            "underlying logic, not just these literal values. Give your final query as <answer>...</answer>.\n\n"
            "## Schema\n```sql\n" + t["schema_text"] + "\n```"
        )

    def step(self, code: str) -> str:
        """Run an exploration query on the original database."""
        obs = execute(self.task["db_path"], code, timeout=4.0, max_rows=20)
        text = format_obs(obs, max_chars=self.config["obs_max_chars"])
        err = None if obs["ok"] else obs["err"]
        if err and err == self.last_err:
            text += SAME_ERROR_HINT
        self.last_err = err
        return text

    def agreement(self, answer: str) -> float:
        """Agreement of the submitted query over the perturbed copies."""
        return score(answer, self.task["gold_sql"], self.task["copies"])
