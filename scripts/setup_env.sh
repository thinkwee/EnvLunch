#!/usr/bin/env bash
# Install the training stack: slime, Megatron-LM and SGLang at the versions used in the paper.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/third_party" && cd "$ROOT/third_party"

[ -d slime ] || git clone https://github.com/THUDM/slime.git
git -C slime checkout 82007faf4b398abd32bd8e07f9638f6cfeb70729
[ -d Megatron-LM ] || git clone https://github.com/NVIDIA/Megatron-LM.git
git -C Megatron-LM checkout 3714d81d418c9f1bca4594fc35f9e8289f652862

pip install "sglang[all]==0.5.11"
pip install -e slime
pip install -r "$ROOT/requirements.txt"
