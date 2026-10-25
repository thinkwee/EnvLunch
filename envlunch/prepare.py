"""Collect the environments named in configs/corpus.yaml from raw downloads into data/envs/{sql,frame}."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LOCATIONS = {
    ("sql", "spider"): "spider/**/database/{name}/{name}.sqlite",
    ("sql", "bird"): "bird/**/train_databases/{name}/{name}.sqlite",
    ("frame", "bird"): "bird/**/train_databases/{name}/{name}.sqlite",
    ("frame", "datamind"): "datamind/**/train_files/{name}",
    ("frame", "dabench"): "dabench/{name}",
}


def locate(raw: Path, pattern: str, name: str) -> Path | None:
    """Find the file of a named environment under the raw download directory."""
    hits = sorted(raw.glob(pattern.format(name=glob.escape(name))))
    return hits[0].resolve() if hits else None


def main():
    """Symlink every listed environment and fail if any is missing."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/envs")
    ap.add_argument("--corpus", default=str(ROOT / "configs" / "corpus.yaml"))
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    with open(args.corpus) as f:
        corpus = yaml.safe_load(f)
    missing = []
    for kind, sources in corpus.items():
        (out / kind).mkdir(parents=True, exist_ok=True)
        for source, names in sources.items():
            for name in names:
                src = locate(raw, LOCATIONS[(kind, source)], name)
                if src is None:
                    missing.append(f"{kind}/{source}/{name}")
                    continue
                dst = out / kind / f"{source}__{src.name}"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(src, dst)
        print(f"{kind}: {len(list((out / kind).iterdir()))} environments")
    if missing:
        print(f"missing {len(missing)}: {missing[:10]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
