#!/usr/bin/env bash
# Export full dataset + query embeddings for one model across several datasets.
# Edit model and datasets below, then run: bash runall.bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"

# model="AutoTimes"
model="TimeMoE"
# datasets=("astro" "deep1b" "F5" "origin" "sald" "seismic")
datasets=("sald" "seismic")
# pretrained: configured backbone + untrained task layers; checkpoint: complete .pth
weights="pretrained"
batch_size=128
output_dir="example/$model/embeddings"
if [[ "$model" == "TimeMoE" || "$model" == "UniTS" ]]; then
  output_dir="example/$model/embeddings_patched_v2"
fi

if [[ "${1:-}" == "--worker" ]]; then
  shift
  for dataset in "$@"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $model on $dataset"
    python -u LLM4SSSsummary_run.py \
      -C "conf/$dataset/$model.json" \
      --weights "$weights" \
      --batch-size "$batch_size" \
      --output-dir "$output_dir"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $model on $dataset"
  done
  echo "All exports completed. Output directory: $output_dir"
  exit 0
fi

nohup bash "$script_dir/runall.bash" --worker "${datasets[@]}" > runall.out 2>&1 &
pid=$!
printf '%s\n' "$pid" > runall.pid
printf 'Batch export started: %s on %s (PID %s). Follow progress: tail -f runall.out\n' \
  "$model" "${datasets[*]}" "$pid"
printf 'Output directory: %s\n' "$output_dir"
