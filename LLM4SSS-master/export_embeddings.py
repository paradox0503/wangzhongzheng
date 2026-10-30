"""Export every dataset/query row as a float32 embedding using the configured model.

Run from the project root:
    python -u export_embeddings.py -C conf/astro/AutoTimes.json --batch-size 128

Export policy:
    - Dataset files: export at most the first 100,000,000 rows.
    - Query files: export all available rows.

The dataset limit applies to every dataset, not only deep1b.

Relative paths in the configuration follow the training script's cwd convention.
Outputs default to example/<model>/embeddings/ with dataset-prefixed filenames.
Existing outputs are refused; select another --output-dir for a new export.
Default pretrained mode uses the configured backbone and untrained task layers.
Use --weights checkpoint to load a complete model_path/example_model.pth instead.
"""

import argparse
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


# Maximum number of rows exported from EVERY dataset file.
# Query files are not subject to this limit.
DEFAULT_MAX_DATASET_ROWS = 100_000_000


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


def row_count(source, width, max_rows=None):
    """Return the number of complete float32 rows available.

    If max_rows is specified, return at most that many rows.

    IMPORTANT:
    The source file is allowed to contain trailing bytes that do not
    constitute a complete row. Only complete rows are considered.
    """

    source = Path(source)
    size = source.stat().st_size
    row_bytes = 4 * width

    if width <= 0:
        raise ValueError(
            f'{source}: width must be positive'
        )

    if size == 0:
        raise ValueError(
            f'{source}: expected nonempty float32 rows '
            f'of width {width}'
        )

    # Number of complete float32 rows in the file.
    #
    # Do NOT require:
    #
    #     size % row_bytes == 0
    #
    # because a large/truncated source file may end in the middle
    # of a row. The exporter only needs the complete rows it will use.
    available_rows = size // row_bytes

    if available_rows == 0:
        raise ValueError(
            f'{source}: contains no complete float32 rows '
            f'of width {width}'
        )

    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError(
                'max_rows must be positive'
            )

        return min(available_rows, max_rows)

    return available_rows


def export_file(
    source,
    target,
    input_dim,
    output_dim,
    batch_size,
    predict,
    max_rows=None,
):
    """Resume a partial output, then publish only a complete validated file."""

    source = Path(source)
    target = Path(target)

    if batch_size <= 0:
        raise ValueError(
            'batch_size must be positive'
        )

    if input_dim <= 0:
        raise ValueError(
            'input_dim must be positive'
        )

    if output_dim <= 0:
        raise ValueError(
            'output_dim must be positive'
        )

    total = row_count(
        source,
        input_dim,
        max_rows=max_rows,
    )

    partial = target.with_name(
        target.name + '.partial'
    )

    expected_size = (
        total
        * output_dim
        * 4
    )

    # ------------------------------------------------------------
    # Existing final output.
    # ------------------------------------------------------------

    if target.exists():

        if target.stat().st_size != expected_size:
            raise ValueError(
                f'{target}: existing output size does not match '
                f'{total} rows'
            )

        print(
            f'{target}: already complete '
            f'({total:,} rows); skipping',
            flush=True,
        )

        return total

    # ------------------------------------------------------------
    # Resume an incomplete output if present.
    # ------------------------------------------------------------

    resume_rows = 0

    if partial.exists():

        partial_size = partial.stat().st_size
        row_bytes = output_dim * 4

        if partial_size % row_bytes:
            raise ValueError(
                f'{partial}: size {partial_size} is not aligned '
                f'to {row_bytes}-byte rows'
            )

        resume_rows = partial_size // row_bytes

        if resume_rows > total:
            raise ValueError(
                f'{partial}: contains {resume_rows} rows, '
                f'exceeding input row count {total}'
            )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = last_report = time.monotonic()

    if resume_rows:
        print(
            f'{source}: resuming at row '
            f'{resume_rows:,}/{total:,} -> {target}',
            flush=True,
        )
    else:
        print(
            f'{source}: {total:,} rows -> {target}',
            flush=True,
        )

    # ------------------------------------------------------------
    # Main export loop.
    # ------------------------------------------------------------

    with source.open('rb') as fin, partial.open(
        'ab' if partial.exists() else 'xb'
    ) as fout:

        # Skip rows that were already exported.
        fin.seek(
            resume_rows
            * input_dim
            * 4
        )

        for start in range(
            resume_rows,
            total,
            batch_size,
        ):

            count = min(
                batch_size,
                total - start,
            )

            # Read count * input_dim float32 values.
            batch = np.fromfile(
                fin,
                dtype=np.float32,
                count=count * input_dim,
            )

            if batch.size != count * input_dim:
                raise ValueError(
                    f'Input truncated at row {start}'
                )

            batch = batch.reshape(
                count,
                input_dim,
            )

            if not np.isfinite(batch).all():
                raise ValueError(
                    f'Nonfinite input in rows '
                    f'{start}:{start + count}'
                )

            # Model inference.
            result = np.asarray(
                predict(batch),
                dtype=np.float32,
            )

            if (
                result.shape
                != (count, output_dim)
                or not np.isfinite(result).all()
            ):
                raise ValueError(
                    f'Invalid model output at row {start}: '
                    f'{result.shape}'
                )

            # Append embedding vectors.
            result.tofile(fout)

            done = start + count

            now = time.monotonic()

            if (
                now - last_report >= 10
                or done == total
            ):

                generated = (
                    done - resume_rows
                )

                rate = generated / max(
                    now - started,
                    0.001,
                )

                eta = (
                    (total - done)
                    / max(rate, 0.001)
                )

                print(
                    f'{target.name}: '
                    f'{done:,}/{total:,} '
                    f'({100 * done / total:.1f}%), '
                    f'{rate:.1f} rows/s, '
                    f'ETA {eta:.0f}s',
                    flush=True,
                )

                last_report = now

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # We intentionally do NOT check EOF here.
        #
        # Dataset export is capped at the first 100,000,000 rows.
        # The source file is expected to contain additional data.
        #
        # Query export also does not need this check because the
        # exact number of complete rows was determined beforehand.
        # --------------------------------------------------------

    # ------------------------------------------------------------
    # Validate output size.
    # ------------------------------------------------------------

    actual_size = partial.stat().st_size

    if actual_size != expected_size:
        raise ValueError(
            f'Output size mismatch: '
            f'expected {expected_size} bytes, '
            f'got {actual_size} bytes'
        )

    # ------------------------------------------------------------
    # Publish atomically.
    # ------------------------------------------------------------

    # Hard-link publication refuses to replace an existing result.
    os.link(
        partial,
        target,
    )

    partial.unlink()

    return total


def main(argv=None):

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            'Supported models: '
            + ', '.join(SUPPORTED_MODELS)
        ),
    )

    parser.add_argument(
        '-C',
        '--conf',
        required=True,
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
    )

    parser.add_argument(
        '--device',
        help='Override config device, e.g. cuda:0',
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
    )

    parser.add_argument(
        '--weights',
        choices=(
            'pretrained',
            'checkpoint',
        ),
        default='pretrained',
        help=(
            'pretrained: configured backbone with '
            'untrained task layers; '
            'checkpoint: complete '
            'model_path/example_model.pth'
        ),
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Initialization seed (default: 42)',
    )

    parser.add_argument(
        '--max-dataset-rows',
        type=int,
        default=DEFAULT_MAX_DATASET_ROWS,
        help=(
            'Maximum number of rows to export from EVERY '
            'dataset file. Query files are unaffected. '
            'Default: 100000000. '
            'Use 0 or a negative value to export all dataset rows.'
        ),
    )

    args = parser.parse_args(argv)

    if args.batch_size <= 0:
        parser.error(
            '--batch-size must be positive'
        )

    if not 0 <= args.seed < 2**32:
        parser.error(
            '--seed must be between 0 and 4294967295'
        )

    # Dataset row limit.
    #
    # Default:
    #     100,000,000
    #
    # <= 0:
    #     unlimited
    if args.max_dataset_rows <= 0:
        max_dataset_rows = None
    else:
        max_dataset_rows = args.max_dataset_rows

    print(
        'Mode: export-only; training disabled.',
        flush=True,
    )

    print(
        f'Exporter: {Path(__file__).resolve()}',
        flush=True,
    )

    from utils.conf import Configuration

    conf = Configuration(
        args.conf
    )

    model_selected = conf.getEntry(
        'model_selected'
    )

    if model_selected == 'MyLLM4SSS9':
        parser.error(
            'MyLLM4SSS9 requires paired inputs; '
            'an independent per-row embedding definition '
            'is needed before full-dataset export.'
        )

    if model_selected not in SUPPORTED_MODELS:
        parser.error(
            f'Unsupported model: {model_selected}. '
            f'Choose from: '
            f'{", ".join(SUPPORTED_MODELS)}'
        )

    if args.device:
        conf.confLoaded['device'] = args.device

    conf.confLoaded['batch_size'] = (
        args.batch_size
    )

    dataset = conf.getEntry(
        'dataset_selected'
    )

    print(
        f'Model: {model_selected}; '
        f'dataset: {dataset}; '
        f'batch size: {args.batch_size}',
        flush=True,
    )

    prefix = (
        'deep1b'
        if dataset == 'deep1b'
        else dataset
    )

    data_dir = Path(
        conf.getEntry('data_path')
    )

    output_dir = (
        args.output_dir
        or Path('example')
        / model_selected
        / 'embeddings'
    )

    checkpoint_path = None

    if args.weights == 'checkpoint':

        checkpoint_path = (
            Path(
                conf.getEntry(
                    'model_path'
                )
            )
            / 'example_model.pth'
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f'Missing checkpoint: '
                f'{checkpoint_path}. '
                f'Checkpoint mode requires complete '
                f'model weights and will not fall back '
                f'to initialization.'
            )

    input_dim = conf.getEntry(
        'len_series'
    )

    output_dim = conf.getEntry(
        'len_reduce'
    )

    dataset_source = (
        data_dir
        / f'{prefix}-dataset.bin'
    )

    query_source = (
        data_dir
        / f'{prefix}-query.bin'
    )

    dataset_target = (
        output_dir
        / f'{dataset}-dataset.bin'
    )

    query_target = (
        output_dir
        / f'{dataset}-query.bin'
    )

    # ------------------------------------------------------------
    # Determine input row counts.
    #
    # ALL DATASETS:
    #     first 100,000,000 rows only
    #
    # ALL QUERIES:
    #     all complete rows
    # ------------------------------------------------------------

    dataset_rows = row_count(
        dataset_source,
        input_dim,
        max_rows=max_dataset_rows,
    )

    query_rows = row_count(
        query_source,
        input_dim,
        max_rows=None,
    )

    print(
        f'Dataset input: first '
        f'{dataset_rows:,} rows '
        f'(limit='
        f'{max_dataset_rows:,}'
        f')',
        flush=True,
    )

    print(
        f'Query input: all '
        f'{query_rows:,} rows',
        flush=True,
    )

    print(
        f'Input dimension: {input_dim}',
        flush=True,
    )

    print(
        f'Output dimension: {output_dim}',
        flush=True,
    )

    import torch

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    device = conf.getEntry(
        'device'
    )

    print(
        f'Weights: {args.weights}; '
        f'initialization seed: {args.seed}; '
        f'device: {device}',
        flush=True,
    )

    if args.weights == 'pretrained':
        print(
            'Using the model\'s configured '
            'backbone weights/initialization. '
            'Task-specific layers are untrained; '
            'no task checkpoint is loaded.',
            flush=True,
        )

    # ------------------------------------------------------------
    # Load only the selected model.
    # ------------------------------------------------------------

    model_class = getattr(
        import_module(
            f'model.{model_selected}'
        ),
        model_selected,
    )

    model = model_class(
        conf
    )

    if checkpoint_path is not None:

        print(
            f'Loading task checkpoint: '
            f'{checkpoint_path}',
            flush=True,
        )

        state = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=True,
        )

        model.load_state_dict(
            state,
            strict=True,
        )

        del state

    model.to(
        device
    ).eval()

    def predict(batch):

        with torch.inference_mode():

            inputs = (
                torch
                .from_numpy(batch)
                .to(device)
            )

            return (
                forward_embeddings(
                    model,
                    model_selected,
                    inputs,
                )
                .float()
                .cpu()
                .numpy()
            )

    # ------------------------------------------------------------
    # 1. Export DATASET
    #
    # ALL datasets:
    #     first max_dataset_rows rows only.
    # ------------------------------------------------------------

    print(
        '\n=== Exporting dataset ===',
        flush=True,
    )

    export_file(
        dataset_source,
        dataset_target,
        input_dim,
        output_dim,
        args.batch_size,
        predict,
        max_rows=max_dataset_rows,
    )

    # ------------------------------------------------------------
    # 2. Export QUERY
    #
    # ALL query rows.
    # ------------------------------------------------------------

    print(
        '\n=== Exporting query ===',
        flush=True,
    )

    export_file(
        query_source,
        query_target,
        input_dim,
        output_dim,
        args.batch_size,
        predict,
        max_rows=None,
    )

    print(
        f'\nExport completed! '
        f'Output directory: '
        f'{output_dir.resolve()}',
        flush=True,
    )


if __name__ == '__main__':
    main()