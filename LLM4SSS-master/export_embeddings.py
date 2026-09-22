"""Export every AutoTimes dataset/query row as a float32 embedding, in order.

Run from the project root:
    python -u export_embeddings.py -C conf/astro/AutoTimes.json --batch-size 128
Relative paths in the configuration follow the training script's cwd convention.
Existing outputs are refused; select another --output-dir for a new export.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


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
    parser = argparse.ArgumentParser(description=__doc__)
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
    if conf.getEntry('model_selected') != 'AutoTimes':
        parser.error('This exporter currently supports AutoTimes only')
    if args.device:
        conf.confLoaded['device'] = args.device
    dataset = conf.getEntry('dataset_selected')
    prefix = 'deep1b' if dataset == 'deep1B' else dataset
    data_dir = Path(conf.getEntry('data_path'))
    output_dir = args.output_dir or Path('example') / 'AutoTimes' / dataset / 'embeddings'
    checkpoint_path = Path(conf.getEntry('model_path')) / 'example_model.pth'
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Missing checkpoint: {checkpoint_path}. Export requires existing weights; '
            'it will not start training or initialize random weights.'
        )
    input_dim = conf.getEntry('len_series')
    output_dim = conf.getEntry('len_reduce')
    jobs = [(data_dir / f'{prefix}-dataset.bin', output_dir / 'dataset_embeddings.bin'),
            (data_dir / f'{prefix}-query.bin', output_dir / 'query_embeddings.bin')]
    for source, target in jobs:
        row_count(source, input_dim)
        if target.exists() or target.with_name(target.name + '.partial').exists():
            raise FileExistsError(f'{target}: choose a new --output-dir')
    metadata_path = output_dir / 'metadata.json'
    if metadata_path.exists():
        raise FileExistsError(metadata_path)

    import torch
    from model.AutoTimes import AutoTimes
    device = conf.getEntry('device')
    print(f'Loading {checkpoint_path} on {device}', flush=True)
    model = AutoTimes(conf)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()

    def predict(batch):
        with torch.inference_mode():
            return model(torch.from_numpy(batch).to(device)).float().cpu().numpy()

    outputs = []
    for source, target in jobs:
        count = export_file(source, target, input_dim, output_dim, args.batch_size, predict)
        outputs.append({'source': str(source.resolve()), 'file': target.name,
                        'shape': [count, output_dim]})
    metadata = {'model': 'AutoTimes', 'dataset': dataset, 'dtype': 'float32',
                'byteorder': sys.byteorder, 'order': 'original input row order',
                'checkpoint': str(checkpoint_path.resolve()),
                'input_dim': input_dim, 'outputs': outputs}
    with metadata_path.open('x', encoding='utf-8') as fout:
        json.dump(metadata, fout, indent=2)
    print(f'Export completed! Metadata: {metadata_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
