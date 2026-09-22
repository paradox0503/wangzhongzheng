#!/usr/bin/env bash
# Full dataset + query inference using the existing AutoTimes checkpoint.
# Edit the settings below, then run: bash run.bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dataset="astro"
batch_size=128
output_dir="example/AutoTimes/$dataset/embeddings_v2"

nohup python -u LLM4SSSsummary_run.py \
  -C "conf/$dataset/AutoTimes.json" \
  --batch-size "$batch_size" \
  --output-dir "$output_dir" > run.out 2>&1 &
pid=$!
printf '%s\n' "$pid" > run.pid
printf 'Export started (PID %s). Follow progress: tail -f run.out\n' "$pid"
printf 'Output directory: %s\n' "$output_dir"
