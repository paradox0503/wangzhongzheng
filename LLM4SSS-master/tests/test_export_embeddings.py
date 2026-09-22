import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'export_embeddings.py'


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'Full-dataset exporter is missing')
        spec = importlib.util.spec_from_file_location('export_embeddings', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'input.bin'
        self.target = Path(self.temp.name) / 'output.bin'
        self.data = np.arange(35, dtype=np.float32).reshape(7, 5)
        self.data.tofile(self.source)

    def test_order_tail_and_float32(self):
        sizes = []
        def predict(batch):
            sizes.append(len(batch))
            return batch[:, :2].astype(np.float64) * 2
        count = self.module.export_file(self.source, self.target, 5, 2, 3, predict)
        self.assertEqual(count, 7)
        self.assertEqual(sizes, [3, 3, 1])
        np.testing.assert_array_equal(np.fromfile(self.target, dtype=np.float32).reshape(7, 2), self.data[:, :2] * 2)

    def test_existing_output_preserved(self):
        self.target.write_bytes(b'existing')
        with self.assertRaises(FileExistsError):
            self.module.export_file(self.source, self.target, 5, 2, 3, lambda x: x[:, :2])
        self.assertEqual(self.target.read_bytes(), b'existing')

    def test_malformed_input_rejected(self):
        self.source.write_bytes(b'bad')
        with self.assertRaises(ValueError):
            self.module.export_file(self.source, self.target, 5, 2, 3, lambda x: x[:, :2])
        self.assertFalse(self.target.exists())

    def test_invalid_prediction_not_published(self):
        with self.assertRaises(ValueError):
            self.module.export_file(self.source, self.target, 5, 2, 3, lambda x: np.full((len(x), 2), np.nan))
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
