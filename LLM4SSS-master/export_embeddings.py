"""Export every dataset/query row as a float32 embedding using the configured model.

Run from the project root:
    python -u export_embeddings.py -C conf/astro/AutoTimes.json --batch-size 128
Relative paths in the configuration follow the training script's cwd convention.
Outputs default to example/<model>/embeddings/ with dataset-prefixed filenames.
Matching outputs resume/skip; changed export identities require a new directory.
Default pretrained mode uses the configured backbone and untrained task layers.
Use --weights checkpoint to load a complete model_path/example_model.pth instead.
"""
import argparse
from contextlib import nullcontext
from importlib import import_module
import os
from pathlib import Path
import random
import time

import numpy as np


SUPPORTED_MODELS = (
    'AutoTimes', 'GPT4SSS', 'TimeLLM', 'UniTime', 'S2IPLLM',
    'TimeMixer', 'UniTS', 'TimeMoE',
) + tuple(f'MyLLM4SSS{i}' for i in range(1, 12) if i != 9)


def forward_embeddings(model, model_name, batch):
    """Adapt the project's model interfaces to one final vector per input row."""
    # MyLLM4SSS7 allocates tensors using a cached training batch size.
    # Update it for every batch, including the final partial batch.
    if hasattr(model, 'batch_size'):
        model.batch_size = batch.shape[0]
    if model_name == 'UniTime':
        return model((batch, batch.new_ones(batch.shape)))
    result = model(batch)
    if model_name == 'MyLLM4SSS2':
        return result[1]  # First output is the intermediate representation.
    return result


def row_count(source, width):
    size = Path(source).stat().st_size
    if width <= 0 or size == 0 or size % (4 * width):
        raise ValueError(f'{source}: expected nonempty float32 rows of width {width}')
    return size // (4 * width)


def export_file(source, target, input_dim, output_dim, batch_size, predict):
    from utils.export_runtime import export_lock
    with export_lock(str(target) + '.lock'):
        return _export_file(source, target, input_dim, output_dim, batch_size, predict)


def _export_file(source, target, input_dim, output_dim, batch_size, predict):
    """Resume a partial output, then publish only a complete validated file."""
    from utils.export_runtime import file_identity
    source, target = Path(source), Path(target)
    source_identity = file_identity(source)
    if batch_size <= 0 or output_dim <= 0:
        raise ValueError('batch_size and output_dim must be positive')
    total = row_count(source, input_dim)
    partial = target.with_name(target.name + '.partial')
    expected_size = total * output_dim * 4
    if target.exists():
        if target.stat().st_size != expected_size:
            raise ValueError(f'{target}: existing output size does not match {total} rows')
        print(f'{target}: already complete ({total:,} rows); skipping', flush=True)
        return total

    resume_rows = 0
    if partial.exists():
        partial_size = partial.stat().st_size
        row_bytes = output_dim * 4
        if partial_size % row_bytes:
            raise ValueError(f'{partial}: size {partial_size} is not aligned to {row_bytes}-byte rows')
        resume_rows = partial_size // row_bytes
        if resume_rows > total:
            raise ValueError(f'{partial}: contains {resume_rows} rows, exceeding input row count {total}')
    target.parent.mkdir(parents=True, exist_ok=True)
    started = last_report = time.monotonic()
    if resume_rows:
        print(f'{source}: resuming at row {resume_rows:,}/{total:,} -> {target}', flush=True)
    else:
        print(f'{source}: {total:,} rows -> {target}', flush=True)
    with source.open('rb') as fin, partial.open('ab' if partial.exists() else 'xb') as fout:
        fin.seek(resume_rows * input_dim * 4)
        for start in range(resume_rows, total, batch_size):
            count = min(batch_size, total - start)
            batch = np.fromfile(fin, dtype=np.float32, count=count * input_dim)
            if batch.size != count * input_dim:
                raise ValueError(f'Input truncated at row {start}')
            batch = batch.reshape(count, input_dim)
            if not np.isfinite(batch).all():
                raise ValueError(f'Nonfinite input in rows {start}:{start + count}')
            result = np.asarray(predict(batch), dtype=np.float32)
            if result.shape != (count, output_dim) or not np.isfinite(result).all():
                raise ValueError(f'Invalid model output at row {start}: {result.shape}')
            result.tofile(fout)
            done = start + count
            now = time.monotonic()
            if now - last_report >= 10 or done == total:
                generated = done - resume_rows
                rate = generated / max(now - started, 0.001)
                print(f'{target.name}: {done:,}/{total:,} ({100 * done / total:.1f}%), '
                      f'{rate:.1f} rows/s, ETA {(total - done) / max(rate, 0.001):.0f}s', flush=True)
                last_report = now
        if fin.read(1):
            raise ValueError('Input size changed during export')
    if partial.stat().st_size != expected_size:
        raise ValueError('Output size mismatch')
    if file_identity(source) != source_identity:
        raise ValueError('Input file changed during export; use a new --output-dir')
    # Hard-link publication is atomic and refuses to replace an existing result.
    os.link(partial, target)
    partial.unlink()
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, epilog='Supported models: ' + ', '.join(SUPPORTED_MODELS))
    parser.add_argument('-C', '--conf', required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', help='Override config device, e.g. cuda:0')
    parser.add_argument('--precision', choices=('auto', 'fp32', 'bf16', 'fp16'),
                        help='Override inference_precision in config (otherwise fp32). '
                             'auto uses BF16 on supported CUDA GPUs, otherwise FP32.')
    parser.add_argument('--attn-implementation', choices=('eager', 'flash_attention_2'),
                        help='TimeMoE only; FlashAttention2 requires a compatible server installation')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--weights', choices=('pretrained', 'checkpoint'), default='pretrained',
                        help='pretrained: configured backbone with untrained task layers; '
                             'checkpoint: complete model_path/example_model.pth')
    parser.add_argument('--seed', type=int, default=42, help='Initialization seed (default: 42)')
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 4294967295')

    print('Mode: export-only; training disabled.', flush=True)
    print(f'Exporter: {Path(__file__).resolve()}', flush=True)
    from utils.conf import Configuration
    conf = Configuration(args.conf)
    model_selected = conf.getEntry('model_selected')
    if model_selected == 'MyLLM4SSS9':
        parser.error('MyLLM4SSS9 requires paired inputs; an independent per-row embedding '
                     'definition is needed before full-dataset export.')
    if model_selected not in SUPPORTED_MODELS:
        parser.error(f'Unsupported model: {model_selected}. Choose from: ' + ', '.join(SUPPORTED_MODELS))
    if args.device:
        conf.confLoaded['device'] = args.device
    if args.attn_implementation:
        if model_selected != 'TimeMoE':
            parser.error('--attn-implementation is only supported for TimeMoE')
        conf.confLoaded['attn_implementation'] = args.attn_implementation
    conf.confLoaded['batch_size'] = args.batch_size
    dataset = conf.getEntry('dataset_selected')
    print(f'Model: {model_selected}; dataset: {dataset}; batch size: {args.batch_size}', flush=True)
    prefix = 'deep1b' if dataset == 'deep1B' else dataset
    data_dir = Path(conf.getEntry('data_path'))
    output_dir = args.output_dir or Path('example') / model_selected / conf.confLoaded.get('export_subdir', 'embeddings')
    checkpoint_path = None
    if args.weights == 'checkpoint':
        checkpoint_path = Path(conf.getEntry('model_path')) / 'example_model.pth'
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Missing checkpoint: {checkpoint_path}. Checkpoint mode requires complete '
                'model weights and will not fall back to initialization.'
            )
    input_dim = conf.getEntry('len_series')
    output_dim = conf.getEntry('len_reduce')
    jobs = [(data_dir / f'{prefix}-dataset.bin', output_dir / f'{dataset}-dataset.bin'),
            (data_dir / f'{prefix}-query.bin', output_dir / f'{dataset}-query.bin')]
    for source, target in jobs:
        row_count(source, input_dim)
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = conf.getEntry('device')
    from utils.export_runtime import resolve_precision, export_identity, ensure_export_identity
    requested_precision = args.precision or conf.confLoaded.get('inference_precision', 'fp32')
    precision = resolve_precision(requested_precision, device, torch)
    if (model_selected == 'TimeMoE'
            and conf.confLoaded.get('attn_implementation', 'eager') == 'flash_attention_2'
            and precision == 'fp32'):
        raise ValueError('TimeMoE FlashAttention2 requires CUDA bf16/fp16; '
                         'use --precision bf16/fp16 or --attn-implementation eager')
    print(f'Inference precision: {precision} (requested: {requested_precision})', flush=True)
    identity = export_identity(conf.confLoaded, jobs, args.weights, args.seed, precision, checkpoint_path)
    ensure_export_identity(output_dir / f'{dataset}-export.json', identity, [target for _, target in jobs])
    if all(target.is_file() and target.stat().st_size == row_count(source, input_dim) * output_dim * 4
           for source, target in jobs):
        print('Dataset and query already complete with matching export identity; skipping model loading.', flush=True)
        return
    print(f'Weights: {args.weights}; initialization seed: {args.seed}; device: {device}', flush=True)
    if args.weights == 'pretrained':
        print('Using the model\'s configured backbone weights/initialization. '
              'Task-specific layers are untrained; no task checkpoint is loaded.', flush=True)
    # Load only the selected model; other models' optional dependencies are irrelevant.
    model_class = getattr(import_module(f'model.{model_selected}'), model_selected)
    model = model_class(conf)
    if model_selected in ('TimeMoE', 'UniTS'):
        prefix_tokens = conf.confLoaded.get('prompt_num', 0) if model_selected == 'UniTS' else 0
        print(f'Backbone tokens per row: {model.patch_num} data + {prefix_tokens} prompt', flush=True)
    if checkpoint_path is not None:
        print(f'Loading task checkpoint: {checkpoint_path}', flush=True)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
    model.to(device).eval()
    backbone_config = getattr(getattr(model, 'llm', None), 'config', None)
    if backbone_config is not None:
        # These outputs are unused by sequence-to-vector adapters.
        backbone_config.use_cache = False
        backbone_config.output_attentions = False
        backbone_config.output_hidden_states = False
    amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[precision] if precision != 'fp32' else None

    def predict(batch):
        amp = torch.autocast(device_type='cuda', dtype=amp_dtype) if amp_dtype is not None else nullcontext()
        with torch.inference_mode(), amp:
            inputs = torch.from_numpy(batch).to(device)
            return forward_embeddings(model, model_selected, inputs).float().cpu().numpy()

    for source, target in jobs:
        if export_identity(conf.confLoaded, jobs, args.weights, args.seed, precision, checkpoint_path) != identity:
            raise ValueError('Inputs/weights/code changed during export; use a new --output-dir')
        export_file(source, target, input_dim, output_dim, args.batch_size, predict)
    print(f'Export completed! Output directory: {output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
