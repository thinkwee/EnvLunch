"""Dataframe warm-up environment."""

from __future__ import annotations

import functools

from envlunch.env import Env
from envlunch.frame.actions import run_gold
from envlunch.frame.artifact import match
from envlunch.frame.relations import load_relations, perturbed_copies
from envlunch.frame.sandbox import run_on_copies, run_stdout

SYSTEM = (
    "You are an expert data-analysis agent.\n\n"
    "Each turn, emit EXACTLY ONE of:\n\n"
    "  <py>python code</py>\n"
    "    Inspect/compute on the relations (read-only). Use pandas on the named\n"
    "    DataFrames, or sql(\"SELECT ...\"). PRINT what you want to see; the system\n"
    "    returns stdout (truncated).\n\n"
    "  <answer>...</answer>\n"
    "    Your FINAL code: assign the computed artifact to a variable `result`\n"
    "    (a scalar, a dict, or a DataFrame). Example: <answer>result = df['fare'].mean()</answer>.\n\n"
    "Sandbox namespace: `pd` (pandas) and `np` (numpy) are ALREADY imported; each relation is "
    "preloaded as a pandas DataFrame by its name; `sql(\"...\")` runs a DuckDB query over the "
    "relations and returns a DataFrame. NEVER write an `import` statement (it errors and wastes a "
    "turn). scipy/sklearn/matplotlib are NOT available.\n\n"
    "Strategy: use <py> to explore and compute, then emit <answer>. By turn {max_turns} you MUST "
    "emit <answer>; unanswered tasks score 0. Committing EARLY once you are confident is rewarded "
    "— do not waste turns.\n"
    "Do not chat. Emit only <py>...</py> or <answer>...</answer>."
)


@functools.lru_cache(maxsize=64)
def _copies(spec: str, k: int, seed: int, frac: float, amp: float):
    """Cached perturbed copies of an environment."""
    return perturbed_copies(load_relations(spec), k, seed, frac, amp)


class FrameEnv(Env):
    """Recover a pandas/SQL program from its output by computing on the tables."""
    name = "frame"
    tag = "py"
    system = SYSTEM

    def user_message(self) -> str:
        """Target output and table schemas."""
        return (
            "## Task\nWrite a pandas/SQL program whose `result` equals the TARGET output below, computed "
            "from the relations. Explore with <py> first, then commit the program as "
            "<answer>result = ...</answer>.\n\n"
            f"## TARGET output\n{self.task['target_repr']}\n\n"
            f"## Relations\n{self.task['schema_text']}"
        )

    def step(self, code: str) -> str:
        """Run exploration code on the original tables."""
        return run_stdout(self.task["spec"], code, self.config["obs_max_chars"])

    def agreement(self, answer: str) -> float:
        """Minimum agreement of the submission with the hidden program over the copies."""
        c = self.cfg
        copies = _copies(self.task["spec"], c["copies"], self.task.get("copy_seed", 0), c["drop_frac"], c["scale_amp"])
        cand = run_on_copies(copies, answer)
        if not copies or cand is None:
            return 0.0
        scores = [match(a, g) for a, g in zip(cand, (run_gold(dict(w), self.task["gold_code"]) for w in copies))
                  if g is not None]
        return min(scores) if scores else 0.0
