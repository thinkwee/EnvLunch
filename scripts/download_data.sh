#!/usr/bin/env bash
# Download the raw warm-up environment sources: Spider, BIRD train, DataMind tables, DABench tables.
set -euo pipefail

RAW=${1:-data/raw}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$RAW"/spider "$RAW"/bird "$RAW"/datamind "$RAW"/dabench

if [ ! -f "$RAW/spider/spider.zip" ]; then
    gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O "$RAW/spider/spider.zip"
fi
unzip -q -n "$RAW/spider/spider.zip" -d "$RAW/spider"

wget -c -q --show-progress https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip -O "$RAW/bird/train.zip"
unzip -q -n "$RAW/bird/train.zip" -d "$RAW/bird"
find "$RAW/bird" -name train_databases.zip | while read -r z; do unzip -q -n "$z" -d "$(dirname "$z")"; done

hf download zjunlp/DataMind-Data rl/train_files.zip --repo-type dataset --local-dir "$RAW/datamind"
unzip -q -n "$RAW/datamind/rl/train_files.zip" -d "$RAW/datamind"

python3 -c "import yaml; print('\n'.join(yaml.safe_load(open('$ROOT/configs/corpus.yaml'))['frame']['dabench']))" |
while read -r name; do
    [ -f "$RAW/dabench/$name" ] || curl -sSfL -o "$RAW/dabench/$name" \
        "https://raw.githubusercontent.com/InfiAgent/InfiAgent/main/examples/DA-Agent/data/da-dev-tables/$name"
done

echo "downloaded to $RAW"
