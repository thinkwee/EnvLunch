"""Warm-up environment interface, config loading and the reward of the paper."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "warmup.yaml"


@functools.lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict:
    """Load the warm-up config (ENVLUNCH_CONFIG or configs/warmup.yaml)."""
    with open(path or os.environ.get("ENVLUNCH_CONFIG", DEFAULT_CONFIG)) as f:
        return yaml.safe_load(f)


class Env:
    """One warm-up task: prompt, one interaction step, and agreement of a recovered trajectory with the hidden one."""
    name = ""
    tag = ""
    system = ""

    def __init__(self, task: dict, config: dict | None = None):
        """Bind a task and the config section of its environment type."""
        self.task = task
        self.config = config or load_config()
        self.cfg = self.config[self.name]

    def system_prompt(self) -> str:
        """System prompt with the turn limit filled in."""
        return self.system.format(max_turns=self.cfg["max_turns"])

    def user_message(self) -> str:
        """Task statement: the environment and the target observation."""
        raise NotImplementedError

    def step(self, code: str) -> str:
        """Execute one exploration action and return the observation."""
        raise NotImplementedError

    def agreement(self, answer: str) -> float:
        """Minimum agreement m over the perturbed copies, in [0, 1]."""
        raise NotImplementedError

    def reward(self, answer: str | None, turns: int) -> float:
        """R = (1 + w (1 - t / t_max)) * min_k m."""
        if not answer:
            return 0.0
        try:
            base = self.agreement(answer)
        except Exception:
            return 0.0
        return base * (1.0 + self.cfg["bonus_weight"] * (1.0 - turns / self.cfg["max_turns"]))


def make_env(task: dict, config: dict | None = None) -> Env:
    """Construct the Env for a task by its task_type."""
    from envlunch.frame.env import FrameEnv
    from envlunch.sql.env import SqlEnv
    return {"sql": SqlEnv, "frame": FrameEnv}[task["task_type"]](task, config)
