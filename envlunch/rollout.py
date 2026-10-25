"""Multi-turn ReAct rollout shared by both environment types (slime generate / reward_func)."""

from __future__ import annotations

import asyncio
import json
import re

from envlunch.env import load_config, make_env

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ENVELOPE = "<|im_end|>\n<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
FINAL_NUDGE = "\n\n[Final turn next — you MUST emit <answer>...</answer>. No more exploration.]"
BAD_FORMAT = "<obs err='emit exactly one <{tag}>...</{tag}> or <answer>...</answer>'/>"
TOO_EARLY = ("<obs err='too early to commit — run at least {n} exploration turns first "
             "(print columns, dtypes, candidate values), then answer'/>")


def parse(text: str, tag: str):
    """Split a model turn into an answer, an exploration action, or malformed output."""
    m = ANSWER_RE.search(text)
    if m:
        return "answer", m.group(1).strip()
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if m:
        return "act", m.group(1).strip()
    return "malformed", ""


def observe(env, turn: int, kind: str, body: str) -> str:
    """Observation returned to the model after a non-final turn."""
    cfg = env.cfg
    if kind == "answer":
        obs = TOO_EARLY.format(n=cfg["min_turns"])
    elif kind == "act":
        obs = env.step(body)
    else:
        obs = BAD_FORMAT.format(tag=env.tag)
    return obs + (FINAL_NUDGE if turn == cfg["max_turns"] - 2 else "")


async def generate(args, sample, sampling_params):
    """Run one multi-turn episode against SGLang and fill the slime Sample."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post
    from slime.utils.types import Sample

    config = load_config()
    tok = GenerateState(args).tokenizer
    env = make_env(json.loads(sample.metadata["task"]), config)
    cfg = env.cfg
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    loop = asyncio.get_event_loop()

    msgs = [{"role": "system", "content": env.system_prompt()}, {"role": "user", "content": env.user_message()}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text, ids, mask, logps = "", [], [], []
    answer, turns, truncated, spin, prev, repeats = None, 0, False, False, None, 0

    for turn in range(cfg["max_turns"]):
        turns = turn + 1
        budget = config["max_response_len"] - len(ids)
        if budget <= 0:
            truncated = True
            break
        out = await post(url, {
            "text": prompt + text,
            "sampling_params": {**sampling_params, "stop": [f"</{env.tag}>", "</answer>"],
                                "max_new_tokens": min(cfg["max_new_tokens"], budget)},
            "return_logprob": True,
        })
        finish = out["meta_info"]["finish_reason"]
        if finish["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample
        cur = out["text"]
        if isinstance(finish.get("matched"), str) and not cur.endswith(finish["matched"]):
            cur += finish["matched"]
        pairs = out["meta_info"].get("output_token_logprobs") or []
        text += cur
        ids += [p[1] for p in pairs]
        mask += [1] * len(pairs)
        logps += [p[0] for p in pairs]

        kind, body = parse(cur, env.tag)
        repeats = repeats + 1 if body and (kind, body) == prev else 1
        spin |= repeats >= config["repeat_limit"]
        prev = (kind, body)
        final = turn == cfg["max_turns"] - 1
        if kind == "answer" and (turn >= cfg["min_turns"] or final):
            answer = body
            break
        if final:
            break
        obs = await loop.run_in_executor(None, observe, env, turn, kind, body)
        env_text = ENVELOPE.format(obs=obs).encode("utf-8", "replace").decode("utf-8")
        env_ids = tok(env_text, add_special_tokens=False)["input_ids"]
        text += env_text
        ids += env_ids
        mask += [0] * len(env_ids)
        logps += [0.0] * len(env_ids)

    sample.prompt = prompt
    sample.tokens = tok(prompt, add_special_tokens=False)["input_ids"] + ids
    sample.response = text
    sample.response_length = len(ids)
    sample.loss_mask = mask
    sample.rollout_log_probs = logps
    sample.status = Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED
    sample.metadata.update({"answer": answer, "turns": turns, "spin": spin})
    return sample


def score(metadata: dict) -> float:
    """Reward of a finished episode from its metadata."""
    if metadata.get("spin"):
        return 0.0
    env = make_env(json.loads(metadata["task"]))
    return env.reward(metadata.get("answer"), int(metadata.get("turns") or env.cfg["max_turns"]))


async def reward_func(args, samples, **kwargs):
    """slime reward hook; scores one sample or a batch."""
    loop = asyncio.get_event_loop()
    if isinstance(samples, list):
        return [await loop.run_in_executor(None, score, s.metadata) for s in samples]
    return await loop.run_in_executor(None, score, samples.metadata)
