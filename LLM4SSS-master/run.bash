#!/usr/bin/env bash
# Full dataset + query inference without task training.
# Edit the settings below, then run: bash run.bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dataset="astro"
model="AutoTimes"
# pretrained: configured backbone + untrained task layers; checkpoint: complete .pth
weights="pretrained"
batch_size=128
output_dir="example/$model/embeddings"

nohup python -u LLM4SSSsummary_run.py \
  -C "conf/$dataset/$model.json" \
  --weights "$weights" \
  --batch-size "$batch_size" \
  --output-dir "$output_dir" > run.out 2>&1 &
pid=$!
printf '%s\n' "$pid" > run.pid
printf 'Export started: %s on %s (PID %s). Follow progress: tail -f run.out\n' "$model" "$dataset" "$pid"
printf 'Output directory: %s\n' "$output_dir"
