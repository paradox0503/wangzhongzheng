# Align TimeMoE with AutoTimes

Goal: correct the point-wise TimeMoE adapter to use the existing AutoTimes
patch encoder, frozen backbone, patch decoder and sequence projection pipeline.
The user requested consistent processing and shorter inference time.

Design: use the same per-dataset win_size, stride, Mlp settings and raw input
as AutoTimes. Feed [B, patches, 384] through the official inputs_embeds API.
Keep all pretrained decoder layers. Remove per-series normalization and the
last-token/statistics head. Do not alter unrelated models or running jobs.
Require non-overlapping patches that cover the entire sequence, including
validating configuration before loading weights. Check context in tokens.

- [x] Add config parity tests and CPU adapter tests with a small differentiable
  backbone fixture (no model downloads), including gradients through frozen
  backbone, partial batches, eval determinism and invalid configurations.
- [x] Replace the adapter and update all six configs. Use a separate task
  checkpoint directory to avoid loading an old incompatible projection.
- [x] Update integration notes with fresh export directory instructions;
  old .partial files must not be resumed with the new architecture.
- [x] Verify syntax, config parity and diff. PyTorch is absent locally; do not
  install dependencies or download weights. Runtime checks and GPU speed/quality
  measurements must be reported as unverified until run in the server environment.

Validation: `python -m unittest discover -s tests -p test_timemoe_patches.py -v`.
The config cases need only the standard library; tensor cases require PyTorch
and otherwise explicitly skip. No training or dataset inference is run here.

Result: the config test initially failed for all six datasets (missing win_size).
After correction it passes; seven tensor tests skip because PyTorch is absent.
AST syntax checks pass for the adapter and tests. No speedup or accuracy claim
has been verified. Existing user edits to run scripts/train.py are untouched.

## Follow-up: UniTS alignment and export latency

The user requested a further review of both adapters and clarified that the
runtime environment is on their server. Do not run local model tests/install
packages. Review source, config, syntax and diffs; leave runnable server checks.

- [x] UniTS uses raw official patch embeddings and position embeddings, retains
  required dynamic-MLP prompt prefix, skips CLS token/head, and projects flattened
  data tokens like GPT4SSS. Partition matches AutoTimes for all six datasets.
- [x] Preserve pretrained decoder structure; require shared UniTS checkpoint
  weights, enforce layout/context/prompt bounds, use a new task-checkpoint path.
- [x] Add export autocast selection (auto=CUDA BF16 if supported, otherwise FP32),
  keep output FP32, disable unused backbone cache/attention/hidden-state outputs.
- [x] Permit explicit official TimeMoE FlashAttention2, reject FP32/CPU export
  combinations; retain eager default, do not claim unsupported SDPA compatibility.
- [x] Add export identity for both dataset/query, refuse incompatible or legacy
  resumes, lock each output against concurrent appends and skip model loading
  when matching exports are complete. Preserve existing user run scripts.
- [x] Prepare server tests for real-source UniTS shapes/gradients, precision,
  resume identity/locks and existing exporter behavior; no new runtime pass claim.
- [x] Complete independent source review and final static checks.

Final static evidence: 10 Python files parsed, 24 JSON configs parsed, six
dataset patch partitions compared, and git diff whitespace checks passed.
Independent reviewer found no remaining important issues after fixes. No tests
were run after the user's server-only environment clarification; actual models,
GPU throughput, memory consumption and retrieval quality remain unverified.
