# UniTS and TimeMoE integration

Goal: add UniTS and TimeMoE-50M to the existing sequence-to-vector experiment structure.

Architecture: one Configuration-based nn.Module per model; input [B, len_series],
output [B, len_reduce]. UniTS uses the official local models/UniTS.py and its
classification feature head. TimeMoE uses the official pretrained causal decoder.
Both have a trainable linear projection and preserve backbone initialization.
Projection inputs include sequence mean and standard deviation alongside learned
features so normalization does not discard level/scale needed by the raw-distance loss.

Constraints: no datasets, package installation, model downloads, training or tests
on this machine, as requested. Review source, configuration and diffs only.

- [x] Add adapters, optional UniTS checkpoint loading, and frozen-backbone handling.
- [x] Register both models in training, FLOP analysis, BSF and coverage exporters.
- [x] Add configs for human, F5, F10, astro, deep1B and sald, using existing data settings.
- [x] Ensure output directories and single-GPU checkpoint saving support the new models.
- [x] Document source/weight setup, defaults and commands; review the final diff.
