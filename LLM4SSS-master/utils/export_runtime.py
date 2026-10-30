"""Inference precision and export identity checks; no eager torch import."""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def export_lock(path):
    """OS-released lock: interruption must not leave a stale ownership claim.

    Keep the tiny lock file after release; unlinking it can create inode races.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f'Another exporter is writing {path}') from exc
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f'Another exporter is writing {path}') from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def resolve_precision(requested, device, torch):
    if requested not in ('auto', 'fp32', 'bf16', 'fp16'):
        raise ValueError('precision must be auto, fp32, bf16 or fp16')
    if requested == 'fp32':
        return 'fp32'
    if not str(device).startswith('cuda'):
        if requested == 'auto':
            return 'fp32'
        raise ValueError('bf16/fp16 export requires a CUDA device; use fp32 on CPU')
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select --device cpu --precision fp32')
    with torch.cuda.device(device):
        supports_bf16 = torch.cuda.is_bf16_supported()
    if requested == 'auto':
        return 'bf16' if supports_bf16 else 'fp32'
    if requested == 'bf16' and not supports_bf16:
        raise ValueError('Selected GPU does not support BF16; select fp32 or fp16')
    return requested


def file_identity(path):
    path = Path(path).expanduser().resolve()
    info = path.stat()
    return dict(path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns)


def export_identity(config, jobs, weights, seed, precision, checkpoint_path):
    # Batch size/device may change on resume; representation settings may not.
    settings = {k: v for k, v in config.items() if k not in ('batch_size', 'device', 'GPUs')}
    project = Path(__file__).resolve().parents[1]
    code = [project / 'model' / (config['model_selected'] + '.py'),
            project / 'model/mlp.py', project / 'model/embed.py',
            project / 'export_embeddings.py', Path(__file__)]
    if config.get('units_repo_path'):
        code.append(Path(config['units_repo_path']).expanduser() / 'models/UniTS.py')
    assets = []
    if checkpoint_path is not None:
        assets.append(file_identity(checkpoint_path))
    if config.get('units_checkpoint'):
        assets.append(file_identity(config['units_checkpoint']))
    # Record local backbone assets without scanning multi-GB tensor contents.
    backbone = config.get('time_moe_path')
    if backbone and not Path(backbone).expanduser().is_dir():
        raise ValueError('Resumable TimeMoE export requires a fixed local model directory; '
                         'download a snapshot and set time_moe_path to that directory')
    if not backbone and config.get('llm_path') and config.get('llm_type'):
        backbone = str(Path(config['llm_path']) / config['llm_type'])
    if backbone and Path(backbone).expanduser().is_dir():
        for path in sorted(Path(backbone).expanduser().iterdir()):
            if path.is_file() and path.suffix in ('.json', '.py', '.bin', '.safetensors', '.pt', '.pth'):
                assets.append(file_identity(path))
    return dict(version=1, config=settings, weights=weights, seed=seed,
                precision=precision, sources=[file_identity(source) for source, _ in jobs],
                assets=assets, code={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in code})


def ensure_export_identity(path, identity, targets):
    """Reject stale/legacy outputs before model loading or appending any rows."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != identity:
            raise ValueError(f'{path}: export settings/weights/code/input changed; use a new --output-dir')
        return
    for target in map(Path, targets):
        if target.exists() or target.with_name(target.name + '.partial').exists():
            raise ValueError(f'{target}: existing output has no export identity; use a new --output-dir')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as fout:
        json.dump(identity, fout, indent=2, sort_keys=True)
        fout.write('\n')
