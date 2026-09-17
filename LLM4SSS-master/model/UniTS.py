"""UniTS adapter for the project's sequence-to-vector distance learning task."""

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from utils.conf import Configuration


class UniTS(nn.Module):
    def __init__(self, conf: Configuration):
        super().__init__()
        self.len_series = conf.getEntry('len_series')
        self.len_reduce = conf.getEntry('len_reduce')
        self.dataset = conf.getEntry('dataset_selected')
        self.task = 'similarity'
        self.freeze_backbone = conf.getEntry('freeze_backbone')
        args = SimpleNamespace(
            d_model=conf.getEntry('dim_model'),
            n_heads=conf.getEntry('head_num'),
            e_layers=conf.getEntry('enc_layers'),
            prompt_num=conf.getEntry('prompt_num'),
            patch_len=conf.getEntry('patch_len'),
            stride=conf.getEntry('stride'),
            dropout=conf.getEntry('dropout'),
        )
        if args.patch_len <= 0 or args.patch_len != args.stride:
            raise ValueError('UniTS requires patch_len == stride > 0.')
        if args.d_model <= 0 or args.n_heads <= 0 or args.d_model % args.n_heads or args.d_model % 8:
            raise ValueError('UniTS dim_model must be positive and divisible by head_num and 8.')
        if args.e_layers <= 0 or args.prompt_num <= 0:
            raise ValueError('UniTS enc_layers and prompt_num must be positive.')

        # Load only the official model file, avoiding its top-level utils package
        # colliding with this project's utils. Extra dependencies are lazy-loaded.
        source = Path(conf.getEntry('units_repo_path')).expanduser() / 'models' / 'UniTS.py'
        if not source.is_file():
            raise FileNotFoundError(f'UniTS source not found: {source}. Set units_repo_path to the official checkout.')
        spec = importlib.util.spec_from_file_location('_llm4sss_official_units', source)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot load UniTS source: {source}')
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        tasks = [(self.task, dict(dataset=self.dataset, task_name='classification',
                                 enc_in=1, num_class=1, seq_len=self.len_series))]
        # The existing experiment initializer skips modules named llm, preserving
        # official initialization and any pretrained weights loaded below.
        self.llm = upstream.Model(args, tasks, pretrain=False)
        checkpoint_path = conf.getEntry('units_checkpoint')
        if checkpoint_path:
            self._load_backbone(checkpoint_path)
        elif self.freeze_backbone:
            raise ValueError('Set units_checkpoint before freezing the UniTS backbone.')
        else:
            logging.info('UniTS: training official architecture from scratch.')
        self.llm.requires_grad_(not self.freeze_backbone)
        if self.freeze_backbone:
            self.llm.eval()
        # Preserve level/scale for the project's distance loss on raw sequences.
        self.projection = nn.Linear(args.d_model + 2, self.len_reduce)

    def _load_backbone(self, path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        own = self.llm.state_dict()
        # Official multitask checkpoints have dataset-specific tokens. New task
        # tokens retain their initialization; shared feature weights must match.
        required = ('patch_embeddings.', 'position_embedding.', 'blocks.', 'cls_head.')
        missing = [key for key, value in own.items() if key.startswith(required)
                   and (key not in state or state[key].shape != value.shape)]
        if missing:
            raise ValueError(f'Incompatible UniTS checkpoint; check dim_model/enc_layers/patch_len. Missing or mismatched: {missing[:8]}')
        compatible = {key: value for key, value in state.items()
                      if key in own and value.shape == own[key].shape}
        result = self.llm.load_state_dict(compatible, strict=False)
        logging.info('UniTS loaded %d tensors; %d task/unused tensors retain initialization.',
                     len(compatible), len(result.missing_keys))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.llm.eval()
        return self

    def forward(self, x):
        # [batch, len_series] -> official classification features -> [batch, len_reduce]
        if x.ndim != 2 or x.shape[1] != self.len_series:
            raise ValueError('UniTS expects [batch_size, len_series].')
        tokens, means, stdev, n_vars, _ = self.llm.tokenize(x.unsqueeze(-1))
        seq_len = tokens.shape[-2]
        prompt = self.llm.prompt_tokens[self.dataset]
        tokens = self.llm.prepare_prompt(tokens, n_vars, prompt,
                                        self.llm.cls_tokens[self.task], 1,
                                        task_name='classification')
        tokens = self.llm.backbone(tokens, prompt.shape[2], seq_len)
        features = self.llm.cls_head(tokens, return_feature=True)
        features = features[:, 0, 0, :]
        statistics = torch.cat((means[:, 0, :], stdev[:, 0, :]), dim=-1)
        return self.projection(torch.cat((features, statistics.to(features.dtype)), dim=-1))
