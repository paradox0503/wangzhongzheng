#!/usr/bin/env bash
# Full dataset + query inference using the selected model's existing checkpoint.
# Edit the settings below, then run: bash run.bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dataset="astro"
model="AutoTimes"
batch_size=128
output_dir="example/$model/embeddings"

nohup python -u LLM4SSSsummary_run.py \
  -C "conf/$dataset/$model.json" \
  --batch-size "$batch_size" \
  --output-dir "$output_dir" > run.out 2>&1 &
pid=$!
printf '%s\n' "$pid" > run.pid
printf 'Export started: %s on %s (PID %s). Follow progress: tail -f run.out\n' "$model" "$dataset" "$pid"
printf 'Output directory: %s\n' "$output_dir"
