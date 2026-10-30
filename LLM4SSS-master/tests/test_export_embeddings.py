import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'export_embeddings.py'


class ArrayBatch(np.ndarray):
    """CPU fixture for the one tensor factory used by the input adapter."""
    def new_ones(self, shape):
        return np.ones(shape, dtype=self.dtype)


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
        with self.assertRaises(ValueError):
            self.module.export_file(self.source, self.target, 5, 2, 3, lambda x: x[:, :2])
        self.assertEqual(self.target.read_bytes(), b'existing')

    def test_complete_output_skips_prediction(self):
        expected = self.data[:, :2].copy()
        expected.tofile(self.target)
        def unexpected(batch):
            raise AssertionError('Complete output must not run inference')
        self.assertEqual(self.module.export_file(self.source, self.target, 5, 2, 3, unexpected), 7)
        np.testing.assert_array_equal(np.fromfile(self.target, dtype=np.float32).reshape(7, 2), expected)

    def test_partial_output_resumes_at_correct_input_row(self):
        partial = self.target.with_name(self.target.name + '.partial')
        self.data[:3, :2].copy().tofile(partial)
        seen = []
        def predict(batch):
            seen.append(batch.copy())
            return batch[:, :2]
        self.module.export_file(self.source, self.target, 5, 2, 3, predict)
        np.testing.assert_array_equal(np.concatenate(seen), self.data[3:])
        np.testing.assert_array_equal(np.fromfile(self.target, dtype=np.float32).reshape(7, 2), self.data[:, :2])
        self.assertFalse(partial.exists())

    def test_input_changed_during_prediction_is_not_published(self):
        def predict(batch):
            info = self.source.stat()
            import os
            os.utime(self.source, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
            return batch[:, :2]
        with self.assertRaisesRegex(ValueError, 'Input file changed'):
            self.module.export_file(self.source, self.target, 5, 2, 3, predict)
        self.assertFalse(self.target.exists())

    def test_malformed_input_rejected(self):
        self.source.write_bytes(b'bad')
        with self.assertRaises(ValueError):
            self.module.export_file(self.source, self.target, 5, 2, 3, lambda x: x[:, :2])
        self.assertFalse(self.target.exists())

    def test_capped_export_keeps_only_requested_prefix(self):
        seen = []
        def predict(batch):
            seen.append(batch.copy())
            return batch[:, :2]
        count = self.module.export_file(self.source, self.target, 5, 2, 3, predict, max_rows=4)
        self.assertEqual(count, 4)
        np.testing.assert_array_equal(np.concatenate(seen), self.data[:4])
        np.testing.assert_array_equal(np.fromfile(self.target, dtype=np.float32).reshape(4, 2), self.data[:4, :2])

    def test_capped_partial_resumes_without_requiring_source_eof(self):
        partial = self.target.with_name(self.target.name + '.partial')
        self.data[:3, :2].copy().tofile(partial)
        count = self.module.export_file(self.source, self.target, 5, 2, 3,
                                        lambda x: x[:, :2], max_rows=5)
        self.assertEqual(count, 5)
        np.testing.assert_array_equal(np.fromfile(self.target, dtype=np.float32).reshape(5, 2), self.data[:5, :2])

    def test_incomplete_final_input_row_is_warned_and_ignored(self):
        with self.source.open('ab') as stream:
            stream.write(b'tail')
        with self.assertWarns(UserWarning):
            count = self.module.export_file(self.source, self.target, 5, 2, 3, lambda x: x[:, :2])
        self.assertEqual(count, 7)
        np.testing.assert_array_equal(np.fromfile(self.target, dtype=np.float32).reshape(7, 2), self.data[:, :2])

    def test_nonpositive_explicit_row_limit_is_rejected(self):
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.module.row_count(self.source, 5, max_rows=limit)

    def test_invalid_prediction_not_published(self):
        with self.assertRaises(ValueError):
            self.module.export_file(self.source, self.target, 5, 2, 3, lambda x: np.full((len(x), 2), np.nan))
        self.assertFalse(self.target.exists())

    def test_single_input_model_exports_in_order(self):
        self.assertTrue(hasattr(self.module, 'forward_embeddings'))
        model = lambda batch: batch[:, :2] * 3
        predict = lambda batch: self.module.forward_embeddings(model, 'GPT4SSS', batch)
        self.module.export_file(self.source, self.target, 5, 2, 3, predict)
        np.testing.assert_array_equal(
            np.fromfile(self.target, dtype=np.float32).reshape(7, 2), self.data[:, :2] * 3)

    def test_unitime_uses_all_observed_mask_including_last_batch(self):
        self.assertTrue(hasattr(self.module, 'forward_embeddings'))
        def model(inputs):
            batch, mask = inputs
            return (batch * mask)[:, :2]
        predict = lambda batch: self.module.forward_embeddings(model, 'UniTime', batch.view(ArrayBatch))
        self.module.export_file(self.source, self.target, 5, 2, 3, predict)
        np.testing.assert_array_equal(
            np.fromfile(self.target, dtype=np.float32).reshape(7, 2), self.data[:, :2])

    def test_two_stage_model_exports_final_embedding(self):
        self.assertTrue(hasattr(self.module, 'forward_embeddings'))
        model = lambda batch: (batch * 100, batch[:, :2])
        result = self.module.forward_embeddings(model, 'MyLLM4SSS2', self.data)
        np.testing.assert_array_equal(result, self.data[:, :2])

    def test_cached_batch_size_matches_partial_batch(self):
        self.assertTrue(hasattr(self.module, 'forward_embeddings'))
        class Model:
            batch_size = 1280
            def __call__(self, batch):
                result = np.zeros((self.batch_size, 2), dtype=np.float32)
                result[:] = batch[:, :2]
                return result
        model = Model()
        predict = lambda batch: self.module.forward_embeddings(model, 'MyLLM4SSS7', batch)
        self.module.export_file(self.source, self.target, 5, 2, 3, predict)
        np.testing.assert_array_equal(
            np.fromfile(self.target, dtype=np.float32).reshape(7, 2), self.data[:, :2])


if __name__ == '__main__':
    unittest.main()
