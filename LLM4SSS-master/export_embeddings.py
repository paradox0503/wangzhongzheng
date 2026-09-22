"""Export every dataset/query row as a float32 embedding using the configured model.

Run from the project root:
    python -u export_embeddings.py -C conf/astro/AutoTimes.json --batch-size 128
Relative paths in the configuration follow the training script's cwd convention.
Outputs default to example/<model>/embeddings/ with dataset-prefixed filenames.
Existing outputs are refused; select another --output-dir for a new export.
"""
import argparse
from importlib import import_module
import os
from pathlib import Path
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
    """Stream batches through predict; publish only a complete validated file."""
    source, target = Path(source), Path(target)
    if batch_size <= 0 or output_dim <= 0:
        raise ValueError('batch_size and output_dim must be positive')
    total = row_count(source, input_dim)
    partial = target.with_name(target.name + '.partial')
    if target.exists() or partial.exists():
        raise FileExistsError(f'Output already exists: {target} or {partial}; use a new output directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    started = last_report = time.monotonic()
    print(f'{source}: {total:,} rows -> {target}', flush=True)
    with source.open('rb') as fin, partial.open('xb') as fout:
        for start in range(0, total, batch_size):
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
                rate = done / max(now - started, 0.001)
                print(f'{target.name}: {done:,}/{total:,} ({100 * done / total:.1f}%), '
                      f'{rate:.1f} rows/s, ETA {(total - done) / rate:.0f}s', flush=True)
                last_report = now
        if fin.read(1):
            raise ValueError('Input size changed during export')
    if partial.stat().st_size != total * output_dim * 4:
        raise ValueError('Output size mismatch')
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
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')

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
    conf.confLoaded['batch_size'] = args.batch_size
    dataset = conf.getEntry('dataset_selected')
    print(f'Model: {model_selected}; dataset: {dataset}; batch size: {args.batch_size}', flush=True)
    prefix = 'deep1b' if dataset == 'deep1B' else dataset
    data_dir = Path(conf.getEntry('data_path'))
    output_dir = args.output_dir or Path('example') / model_selected / 'embeddings'
    checkpoint_path = Path(conf.getEntry('model_path')) / 'example_model.pth'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Missing checkpoint: {checkpoint_path}. Export requires existing weights; '
            'it will not start training or initialize random weights.'
        )
    input_dim = conf.getEntry('len_series')
    output_dim = conf.getEntry('len_reduce')
    jobs = [(data_dir / f'{prefix}-dataset.bin', output_dir / f'{dataset}-dataset.bin'),
            (data_dir / f'{prefix}-query.bin', output_dir / f'{dataset}-query.bin')]
    for source, target in jobs:
        row_count(source, input_dim)
        if target.exists() or target.with_name(target.name + '.partial').exists():
            raise FileExistsError(f'{target}: choose a new --output-dir')
    import torch
    device = conf.getEntry('device')
    print(f'Loading {checkpoint_path} on {device}', flush=True)
    # Load only the selected model; other models' optional dependencies are irrelevant.
    model_class = getattr(import_module(f'model.{model_selected}'), model_selected)
    model = model_class(conf)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()

    def predict(batch):
        with torch.inference_mode():
            inputs = torch.from_numpy(batch).to(device)
            return forward_embeddings(model, model_selected, inputs).float().cpu().numpy()

    for source, target in jobs:
        export_file(source, target, input_dim, output_dim, args.batch_size, predict)
    print(f'Export completed! Output directory: {output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
