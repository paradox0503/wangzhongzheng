"""Regression: the original command must never enter the training module."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT / 'LLM4SSSsummary_run.py'
NO_TRAINING = '''
import runpy
import sys
from pathlib import Path

class RejectTrainingImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'utils.expe':
            raise AssertionError('Export entry must not import the training module')

sys.meta_path.insert(0, RejectTrainingImport())
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0], run_name='__main__')
'''


class ExportEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.conf = self.root / 'config.json'
        self.checkpoint = self.root / 'missing_checkpoint' / 'example_model.pth'
        self.conf.write_text(json.dumps({
            'model_selected': 'AutoTimes',
            'dataset_selected': 'astro',
            'epoch_max': 100,
            'data_path': str(self.root / 'data'),
            'model_path': str(self.checkpoint.parent),
        }), encoding='utf-8')

    def run_entrypoint(self, *args):
        return subprocess.run(
            [sys.executable, '-B', '-c', NO_TRAINING, str(ENTRYPOINT),
             '-C', str(self.conf), *args],
            cwd=self.root, capture_output=True, text=True, timeout=30,
        )

    def test_old_command_with_100_epochs_checks_checkpoint_without_training(self):
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileNotFoundError', result.stderr)
        self.assertIn(str(self.checkpoint), result.stderr)
        self.assertFalse((self.root / 'example').exists())

    def test_old_entrypoint_accepts_export_options_and_validates_batch_size(self):
        result = self.run_entrypoint('--batch-size', '0', '--output-dir', str(self.root / 'out'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--batch-size must be positive', result.stderr)

    def test_other_models_check_their_checkpoint_without_training(self):
        config = json.loads(self.conf.read_text(encoding='utf-8'))
        for name in ('GPT4SSS', 'TimeLLM', 'UniTime', 'S2IPLLM',
                     'TimeMixer', 'UniTS', 'TimeMoE', 'MyLLM4SSS2'):
            with self.subTest(model=name):
                checkpoint = self.root / name / 'example_model.pth'
                config['model_selected'] = name
                config['model_path'] = str(checkpoint.parent)
                self.conf.write_text(json.dumps(config), encoding='utf-8')
                result = self.run_entrypoint()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('FileNotFoundError', result.stderr)
                self.assertIn(str(checkpoint), result.stderr)

    def test_pair_model_requires_an_explicit_embedding_definition(self):
        config = json.loads(self.conf.read_text(encoding='utf-8'))
        config['model_selected'] = 'MyLLM4SSS9'
        self.conf.write_text(json.dumps(config), encoding='utf-8')
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('paired inputs', result.stderr)


if __name__ == '__main__':
    unittest.main()
