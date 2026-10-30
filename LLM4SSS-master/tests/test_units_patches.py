"""Run on the server with PyTorch, timm and the official UniTS checkout."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get('UNITS_REPO_PATH', str(PROJECT.parent / 'model/UniTS')))
try:
    import torch
except ImportError:
    torch = None


class UniTSConfigTests(unittest.TestCase):
    def test_patch_partition_matches_autotimes(self):
        for path in sorted((PROJECT / 'conf').glob('*/UniTS.json')):
            with self.subTest(dataset=path.parent.name):
                settings = json.loads(path.read_text())
                baseline = json.loads((path.parent / 'AutoTimes.json').read_text())
                self.assertEqual(settings['patch_len'], baseline['win_size'])
                self.assertEqual(settings['stride'], baseline['stride'])
                self.assertEqual(settings['len_series'], baseline['len_series'])
                self.assertEqual(settings['dataset_selected'], baseline['dataset_selected'])
                self.assertIn('model_patched', settings['model_path'])


@unittest.skipUnless(torch is not None and (UPSTREAM / 'models/UniTS.py').is_file(),
                     'Requires server PyTorch/timm and UNITS_REPO_PATH')
class OfficialUniTSAdapterTests(unittest.TestCase):
    def make_model(self, length=128, window=8, **overrides):
        from model.UniTS import UniTS
        settings = dict(len_series=length, len_reduce=16, dataset_selected='test',
                        freeze_backbone=False, dim_model=16, head_num=4, enc_layers=1,
                        prompt_num=10, patch_len=window, stride=window, dropout=0.1,
                        units_repo_path=str(UPSTREAM), units_checkpoint='')
        settings.update(overrides)
        return UniTS(SimpleNamespace(getEntry=settings.__getitem__))

    def test_raw_patches_no_cls_head_and_partial_batches(self):
        for length, window in ((96, 8), (128, 8), (256, 16)):
            model = self.make_model(length, window).eval()
            raw, shapes = [], []
            first = model.llm.patch_embeddings.register_forward_pre_hook(
                lambda module, args: raw.append(args[0].clone()))
            second = model.llm.blocks[0].register_forward_pre_hook(
                lambda module, args: shapes.append(tuple(args[0].shape)))
            try:
                with patch.object(model.llm, 'tokenize', side_effect=AssertionError('No normalization')), \
                     patch.object(model.llm.cls_head, 'forward', side_effect=AssertionError('No CLS head')):
                    for batch in (3, 1):
                        inputs = torch.arange(batch * length, dtype=torch.float32).reshape(batch, length)
                        with torch.inference_mode():
                            result = model(inputs)
                            repeated = model(inputs)
                        self.assertEqual(result.shape, (batch, 16))
                        self.assertTrue(torch.isfinite(result).all())
                        torch.testing.assert_close(result, repeated)
                        torch.testing.assert_close(raw[-1], inputs.unsqueeze(1))
                        self.assertEqual(shapes[-1], (batch, 1, 10 + length // window, 16))
            finally:
                first.remove()
                second.remove()

    def test_training_and_frozen_checkpoint_without_unused_cls_head(self):
        model = self.make_model().train()
        model(torch.randn(2, 128)).square().mean().backward()
        self.assertIsNotNone(model.llm.patch_embeddings.value_embedding.weight.grad)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'units.pth'
            state = {k: v for k, v in model.llm.state_dict().items() if not k.startswith('cls_head.')}
            torch.save(state, checkpoint)
            frozen = self.make_model(units_checkpoint=str(checkpoint), freeze_backbone=True).train()
            self.assertFalse(frozen.llm.training)
            frozen(torch.randn(2, 128)).square().mean().backward()
            self.assertIsNotNone(frozen.projection.weight.grad)
            self.assertTrue(all(p.grad is None for p in frozen.llm.parameters()))

    def test_invalid_layout_is_rejected_before_source_loading(self):
        for settings in (dict(length=127), dict(window=256), dict(stride=4), dict(prompt_num=128)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.make_model(**settings)


if __name__ == '__main__':
    unittest.main()
