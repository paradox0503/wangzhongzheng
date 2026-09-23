"""Exercise export I/O with a small NumPy model instead of unavailable GPU weights."""
from contextlib import nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


class ArrayTensor(np.ndarray):
    def to(self, device):
        return self

    def float(self):
        return self.astype(np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class SmallModel:
    def __init__(self, conf):
        self.weight = np.random.normal(size=(5, 2)).astype(np.float32)
        self.training = True

    def load_state_dict(self, state, strict):
        self.weight = state['weight']

    def to(self, device):
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, batch):
        if self.training:
            raise AssertionError('Export must put the model in eval mode')
        return (batch @ self.weight).view(ArrayTensor)


class ExportWeightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = np.arange(35, dtype=np.float32).reshape(7, 5)
        self.data.tofile(self.root / 'astro-dataset.bin')
        self.data[[0, 2, 6]].tofile(self.root / 'astro-query.bin')
        self.checkpoint = self.root / 'example_model.pth'
        self.conf = self.root / 'config.json'
        self.conf.write_text(json.dumps({
            'model_selected': 'AutoTimes', 'dataset_selected': 'astro',
            'epoch_max': 100, 'data_path': str(self.root),
            'model_path': str(self.root), 'device': 'cpu',
            'len_series': 5, 'len_reduce': 2,
        }), encoding='utf-8')
        spec = importlib.util.spec_from_file_location('export_weights_subject', PROJECT / 'export_embeddings.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.runtime = SimpleNamespace(
            manual_seed=lambda seed: None,
            from_numpy=lambda data: data.view(ArrayTensor),
            inference_mode=nullcontext,
            load=lambda path, **kwargs: {'weight': np.fromfile(path, dtype=np.float32).reshape(5, 2)},
        )

    def export(self, name, mode):
        output = self.root / name
        with patch.dict(sys.modules, {'torch': self.runtime}), \
             patch.object(sys, 'path', [str(PROJECT), *sys.path]), \
             patch.object(self.module, 'import_module', return_value=SimpleNamespace(AutoTimes=SmallModel)), \
             redirect_stdout(io.StringIO()):
            try:
                self.module.main(['-C', str(self.conf), '--weights', mode,
                                  '--batch-size', '3', '--output-dir', str(output)])
            except (SystemExit, FileNotFoundError) as exc:
                self.fail(f'Export in {mode} mode failed: {exc}')
        return output

    def test_pretrained_mode_exports_both_files_without_a_task_checkpoint(self):
        output = self.export('pretrained', 'pretrained')
        self.assertFalse(self.checkpoint.exists())
        self.assertEqual({p.name for p in output.iterdir()}, {'astro-dataset.bin', 'astro-query.bin'})
        dataset = np.fromfile(output / 'astro-dataset.bin', dtype=np.float32).reshape(7, 2)
        query = np.fromfile(output / 'astro-query.bin', dtype=np.float32).reshape(3, 2)
        np.testing.assert_array_equal(query, dataset[[0, 2, 6]])

    def test_checkpoint_mode_applies_saved_weights(self):
        np.eye(5, 2, dtype=np.float32).tofile(self.checkpoint)
        output = self.export('checkpoint', 'checkpoint')
        result = np.fromfile(output / 'astro-dataset.bin', dtype=np.float32).reshape(7, 2)
        np.testing.assert_array_equal(result, self.data[:, :2])

    def test_pretrained_mode_keeps_same_seed_and_ignores_existing_task_weights(self):
        first = self.export('first', 'pretrained')
        np.eye(5, 2, dtype=np.float32).tofile(self.checkpoint)
        second = self.export('second', 'pretrained')
        self.assertEqual((first / 'astro-dataset.bin').read_bytes(),
                         (second / 'astro-dataset.bin').read_bytes())
        self.assertEqual((first / 'astro-query.bin').read_bytes(),
                         (second / 'astro-query.bin').read_bytes())


if __name__ == '__main__':
    unittest.main()
