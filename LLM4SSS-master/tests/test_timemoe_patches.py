"""Patch parity plus CPU adapter checks; no pretrained weights are downloaded."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
try:
    import torch
    from torch import nn
except ImportError:
    torch = None


class ConfigParityTests(unittest.TestCase):
    def test_all_datasets_use_autotimes_patch_settings(self):
        for path in sorted((PROJECT / 'conf').glob('*/TimeMoE.json')):
            with self.subTest(dataset=path.parent.name):
                actual = json.loads(path.read_text())
                baseline = json.loads((path.parent / 'AutoTimes.json').read_text())
                for key in ('len_series', 'win_size', 'stride', 'mlp_dim',
                            'mlp_layers', 'activation', 'dropout'):
                    self.assertEqual(actual.get(key), baseline[key], key)
                self.assertEqual(actual['len_series'] % actual['win_size'], 0)
                self.assertEqual(actual['dataset_selected'], baseline['dataset_selected'])
                self.assertIn('model_patched', actual['model_path'])


@unittest.skipIf(torch is None, 'PyTorch is not installed; run tensor checks on the server')
class AdapterTests(unittest.TestCase):
    def make_model(self, length=128, window=8, freeze=True, **overrides):
        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(use_cache=True)
                self.layer = nn.Linear(384, 384)
                self.calls = []

            def forward(self, *, inputs_embeds, use_cache, return_dict,
                        output_attentions=False, output_hidden_states=False):
                self.calls.append(tuple(inputs_embeds.shape))
                assert use_cache is False and return_dict is True
                return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds))

        settings = dict(len_series=length, len_reduce=16, win_size=window,
                        stride=window, mlp_dim=32, mlp_layers=2, activation='tanh',
                        dropout=0.1, freeze_backbone=freeze, time_moe_path='unused',
                        local_files_only=True)
        settings.update(overrides)
        config = SimpleNamespace(model_type='time_moe', hidden_size=384,
                                 num_hidden_layers=12, num_experts=8,
                                 input_size=1, use_dense=False, max_position_embeddings=4096)
        backbone = Backbone()
        loaders = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *a, **k: SimpleNamespace(model=backbone)))
        with patch.dict(sys.modules, {'transformers': loaders}), \
                patch.object(sys, 'path', [str(PROJECT), *sys.path]):
            from model.TimeMoE import TimeMoE
            model = TimeMoE(SimpleNamespace(getEntry=settings.__getitem__))
        return model

    def test_token_count_output_shape_and_partial_batch(self):
        for length, window in ((96, 8), (128, 8), (256, 16)):
            model = self.make_model(length, window).eval()
            for batch in (3, 1):
                with torch.inference_mode():
                    result = model(torch.randn(batch, length))
                self.assertEqual(result.shape, (batch, 16))
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(model.llm.calls[-1], (batch, length // window, 384))

    def test_frozen_backbone_still_transmits_gradients_to_patch_encoder(self):
        torch.manual_seed(17)
        model = self.make_model().train()
        self.assertFalse(model.llm.training)
        model(torch.randn(3, 128)).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.llm.parameters()))
        for module in (model.encoder, model.decoder, model.projection):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in module.parameters()))

    def test_unfrozen_backbone_gets_gradients(self):
        model = self.make_model(freeze=False).train()
        self.assertTrue(model.llm.training)
        model(torch.randn(2, 128)).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.llm.parameters()))

    def test_eval_is_deterministic_and_raw_patch_values_are_preserved(self):
        model = self.make_model().eval()
        inputs = torch.arange(128, dtype=torch.float32).unsqueeze(0)
        seen = []
        handle = model.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            with torch.inference_mode():
                first, second = model(inputs), model(inputs)
            torch.testing.assert_close(first, second)
            torch.testing.assert_close(seen[0], inputs.reshape(1, 16, 8))
        finally:
            handle.remove()

    def test_invalid_patch_layouts_rejected(self):
        for overrides in ({'stride': 4}, {'length': 127}, {'window': 0}, {'window': 256}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.make_model(**overrides)

    def test_context_limit_is_in_patch_tokens(self):
        model = self.make_model(length=8192, window=16)
        self.assertEqual(model.patch_num, 512)
        with self.assertRaises(ValueError):
            self.make_model(length=65552, window=16)

    def test_wrong_input_shape_rejected(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 127))


if __name__ == '__main__':
    unittest.main()
