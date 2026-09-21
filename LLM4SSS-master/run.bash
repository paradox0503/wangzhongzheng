#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

dataset="${1:-astro}"
# Run sequentially with zero training epochs. JSON files remain unchanged.
for model in AutoTimes GPT4SSS UniTS TimeMoE; do
    echo "Running $model on $dataset (epoch_max=0)"
    python -u - "conf/$dataset/$model.json" > "run_${model}_${dataset}.out" 2>&1 <<'PY'
import os
import sys
from utils.conf import Configuration
from utils.expe import Experiment

conf = Configuration(sys.argv[1])
conf.confLoaded["epoch_max"] = 0
for key in ("log_path", "train_path", "val_path", "test_path",
            "train_indices_path", "val_indices_path", "test_indices_path"):
    os.makedirs(os.path.dirname(conf.getEntry(key)) or ".", exist_ok=True)
os.makedirs(conf.getEntry("model_path"), exist_ok=True)
Experiment(conf).run()
PY
done
