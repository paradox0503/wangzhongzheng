"""Small server-side checks for precision and safe inference resume."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from utils.export_runtime import ensure_export_identity, export_identity, export_lock, resolve_precision


class PrecisionTests(unittest.TestCase):
    def runtime(self, bf16):
        return SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, device=lambda device: nullcontext(),
            is_bf16_supported=lambda: bf16))

    def test_auto_cpu_does_not_require_cuda(self):
        self.assertEqual(resolve_precision('auto', 'cpu', None), 'fp32')

    def test_auto_cuda_only_uses_supported_bf16(self):
        self.assertEqual(resolve_precision('auto', 'cuda:0', self.runtime(True)), 'bf16')
        self.assertEqual(resolve_precision('auto', 'cuda:0', self.runtime(False)), 'fp32')

    def test_explicit_unsupported_precision_rejected(self):
        with self.assertRaises(ValueError):
            resolve_precision('bf16', 'cuda:0', self.runtime(False))
        with self.assertRaises(ValueError):
            resolve_precision('fp16', 'cpu', None)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target = self.root / 'dataset.bin'
        self.manifest = self.root / 'export.json'
        self.identity = dict(seed=42, precision='fp32')

    def test_matching_identity_can_resume(self):
        ensure_export_identity(self.manifest, self.identity, [self.target])
        self.target.with_suffix('.bin.partial').write_bytes(b'partial')
        ensure_export_identity(self.manifest, self.identity, [self.target])

    def test_changed_seed_or_resolved_precision_refused(self):
        ensure_export_identity(self.manifest, self.identity, [self.target])
        for changed in (dict(seed=43, precision='fp32'), dict(seed=42, precision='bf16')):
            with self.assertRaises(ValueError):
                ensure_export_identity(self.manifest, changed, [self.target])

    def test_legacy_complete_and_partial_outputs_refused(self):
        for target in (self.target, self.target.with_suffix('.bin.partial')):
            with self.subTest(target=target):
                target.write_bytes(b'old')
                with self.assertRaises(ValueError):
                    ensure_export_identity(self.manifest, self.identity, [self.target])
                self.assertEqual(target.read_bytes(), b'old')
                target.unlink()

    def test_lock_rejects_second_writer_and_releases(self):
        lock = self.root / 'output.lock'
        with export_lock(lock):
            with self.assertRaises(RuntimeError):
                with export_lock(lock):
                    self.fail('Concurrent writer acquired lock')
        with export_lock(lock):
            pass

    def test_identity_covers_inputs_weights_and_config_but_allows_batch_change(self):
        source = self.root / 'source.bin'
        source.write_bytes(b'input')
        checkpoint = self.root / 'weights.pth'
        checkpoint.write_bytes(b'weights')
        config = dict(model_selected='TimeMoE', win_size=8, batch_size=128, device='cpu')
        def identity(settings=config):
            return export_identity(settings, [(source, self.target)], 'checkpoint', 42, 'fp32', checkpoint)
        first = identity()
        self.assertEqual(first, identity(dict(config, batch_size=512, device='cuda:0')))
        self.assertNotEqual(first, identity(dict(config, win_size=16)))
        source.write_bytes(b'changed input')
        self.assertNotEqual(first, identity())
        second = identity()
        checkpoint.write_bytes(b'changed weights')
        self.assertNotEqual(second, identity())

    def test_remote_unpinned_timemoe_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'fixed local model directory'):
            export_identity(dict(model_selected='TimeMoE', time_moe_path='Maple728/TimeMoE-50M'),
                            [], 'pretrained', 42, 'fp32', None)


if __name__ == '__main__':
    unittest.main()
