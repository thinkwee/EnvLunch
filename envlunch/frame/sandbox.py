"""Subprocess sandbox for model code with pandas, numpy and DuckDB, memory and time limits."""

from __future__ import annotations

import builtins
import multiprocessing as mp
import os
import queue
import tempfile
import threading

MEM_LIMIT = int(os.environ.get("ENVLUNCH_SANDBOX_MEM_GB", "48")) * 1024 ** 3
TIMEOUT = float(os.environ.get("ENVLUNCH_SANDBOX_TIMEOUT", "8"))
N_WORKERS = int(os.environ.get("ENVLUNCH_SANDBOX_WORKERS", "32"))
RECYCLE = int(os.environ.get("ENVLUNCH_SANDBOX_RECYCLE", "300"))

SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "print", "len", "range", "min", "max", "sum", "abs", "round", "sorted", "reversed", "float", "int",
    "list", "dict", "set", "tuple", "str", "bool", "enumerate", "zip", "map", "filter", "any", "all",
    "isinstance", "type", "format", "repr", "divmod", "pow")}
SAFE_BUILTINS.update({"True": True, "False": False, "None": None})
IMPORT_ALLOWED = {"pandas", "numpy", "math", "statistics", "datetime", "collections", "re", "json",
                  "itertools", "functools", "decimal", "fractions", "scipy", "sklearn"}

_CTX = mp.get_context("forkserver")
_CTX.set_forkserver_preload(["numpy", "pandas", "duckdb", "envlunch.frame.artifact", "envlunch.frame.relations"])


def _safe_import(name, *a, **k):
    """Import restricted to an allow-list."""
    if name.split(".")[0] in IMPORT_ALLOWED:
        return __import__(name, *a, **k)
    raise ImportError(f"import of '{name}' is not allowed in the sandbox")


def _namespace(frames: dict, con) -> dict:
    """Sandbox namespace: tables, pd, np and sql()."""
    import numpy as np
    import pandas as pd
    for name, df in frames.items():
        con.register(name, df)
    ns = {"__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import}, "pd": pd, "np": np,
          "sql": lambda query: con.execute(query).df(), **frames}
    if "df" not in ns and len(frames) == 1:
        ns["df"] = next(iter(frames.values()))
    return ns


def _duckdb():
    """Single-threaded DuckDB connection."""
    import duckdb
    try:
        return duckdb.connect(config={"threads": 1, "memory_limit": "2GB"})
    except Exception:
        return duckdb.connect()


def _worker(conn, mem_limit):
    """Worker loop: run exploration code (stdout) or a submission on copies (results)."""
    import contextlib
    import io
    import resource
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    except Exception:
        pass
    os.chdir(tempfile.mkdtemp(prefix="envlunch_sandbox_"))
    from envlunch.frame.artifact import normalize
    from envlunch.frame.relations import load_relations

    cache, served = {}, 0
    conn.send("ready")
    while True:
        try:
            req = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if req is None:
            break
        served += 1
        try:
            if req[0] == "stdout":
                _, spec, code, max_chars = req
                if spec not in cache:
                    if len(cache) > 8:
                        cache.clear()
                    try:
                        cache[spec] = load_relations(spec)
                    except Exception:
                        cache[spec] = None
                rels = cache[spec]
                if rels is None or not rels.tables:
                    conn.send("<obs err='could not load the relation(s)'/>")
                    continue
                con, buf = _duckdb(), io.StringIO()
                try:
                    ns = _namespace({n: d.copy() for n, d in rels.tables.items()}, con)
                    with contextlib.redirect_stdout(buf):
                        exec(code, ns)
                    out = buf.getvalue() or "(no output — did you print()?)"
                except MemoryError:
                    out = buf.getvalue() + "\nError: MemoryError (exceeded sandbox memory limit)"
                except Exception as e:
                    out = buf.getvalue() + f"\nError: {repr(e)[:200]}"
                finally:
                    con.close()
                conn.send(out[:max_chars])
            elif req[0] == "copies":
                _, copies, code = req
                outs = []
                for copy in copies:
                    con = _duckdb()
                    try:
                        ns = _namespace(dict(copy), con)
                        exec(code, ns)
                        outs.append(normalize(ns.get("result")))
                    except Exception:
                        outs.append(None)
                    finally:
                        con.close()
                conn.send(("ok", outs))
        except Exception as e:
            try:
                conn.send(("err", repr(e)[:120]))
            except Exception:
                pass
        if served >= RECYCLE:
            break


class Pool:
    """Pool of sandbox workers, replaced on timeout or crash."""
    def __init__(self, n: int, target=None, mem_limit: int = MEM_LIMIT, timeout: float = TIMEOUT):
        """Start n workers."""
        self.target, self.mem_limit, self.timeout = target or _worker, mem_limit, timeout
        self.idle = queue.Queue()
        for _ in range(n):
            self.idle.put(self._spawn())

    def _spawn(self):
        """Start one worker and wait until it is ready."""
        parent, child = _CTX.Pipe()
        p = _CTX.Process(target=self.target, args=(child, self.mem_limit), daemon=True)
        p.start()
        child.close()
        parent.recv()
        return p, parent

    def _replace(self, p):
        """Kill a worker and start a new one."""
        try:
            p.kill()
            p.join(timeout=2)
        except Exception:
            pass
        self.idle.put(self._spawn())

    def call(self, req, fallback):
        """Send a request to an idle worker; return fallback on timeout."""
        for attempt in (0, 1):
            p, conn = self.idle.get()
            try:
                conn.send(req)
            except (EOFError, BrokenPipeError, OSError):
                self._replace(p)
                continue
            if not conn.poll(self.timeout):
                self._replace(p)
                return fallback
            try:
                res = conn.recv()
            except (EOFError, OSError):
                self._replace(p)
                continue
            self.idle.put((p, conn))
            return res
        return fallback


_POOL = None
_LOCK = threading.Lock()


def _pool() -> Pool:
    """Process-wide worker pool."""
    global _POOL
    if _POOL is None:
        with _LOCK:
            if _POOL is None:
                _POOL = Pool(N_WORKERS)
    return _POOL


def run_stdout(spec: str, code: str, max_chars: int = 600) -> str:
    """Run exploration code on the original tables and return its output."""
    if not code or not code.strip():
        return "(no code)"
    r = _pool().call(("stdout", spec, code, max_chars), "(timeout)")
    return r if isinstance(r, str) else "(timeout)"


def run_on_copies(copies: list, code: str):
    """Run a submission on each perturbed copy and return the results."""
    if not code or not code.strip() or not copies:
        return None
    r = _pool().call(("copies", tuple(copies), code), None)
    return r[1] if isinstance(r, tuple) and r and r[0] == "ok" else None
