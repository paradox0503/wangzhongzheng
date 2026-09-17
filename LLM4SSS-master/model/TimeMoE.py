"""TimeMoE-50M pretrained backbone with a trainable distance-learning head."""

import torch
from torch import nn

from utils.conf import Configuration


class TimeMoE(nn.Module):
    def __init__(self, conf: Configuration):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM

        self.len_series = conf.getEntry('len_series')
        self.len_reduce = conf.getEntry('len_reduce')
        self.freeze_backbone = conf.getEntry('freeze_backbone')
        source = conf.getEntry('time_moe_path')
        local_only = conf.getEntry('local_files_only')
        config = AutoConfig.from_pretrained(source, trust_remote_code=True,
                                            local_files_only=local_only)
        expected = dict(model_type='time_moe', hidden_size=384, num_hidden_layers=12,
                        num_experts=8, input_size=1, use_dense=False)
        if any(getattr(config, key, None) != value for key, value in expected.items()):
            raise ValueError('TimeMoE requires the Maple728/TimeMoE-50M base checkpoint.')
        if self.len_series > config.max_position_embeddings:
            raise ValueError('len_series exceeds the TimeMoE context length.')
        pretrained = AutoModelForCausalLM.from_pretrained(
            source, config=config, trust_remote_code=True, local_files_only=local_only,
            torch_dtype=torch.float32, attn_implementation='eager',
        )
        # Use hidden features, not forecast logits or autoregressive generation.
        # Naming matches the existing initializer's pretrained-module exclusion.
        self.llm = pretrained.model
        self.llm.config.use_cache = False
        self.llm.requires_grad_(not self.freeze_backbone)
        if self.freeze_backbone:
            self.llm.eval()
        self.projection = nn.Linear(config.hidden_size + 2, self.len_reduce)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.llm.eval()
        return self

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.len_series:
            raise ValueError('TimeMoE expects [batch_size, len_series].')
        mean = x.mean(dim=1, keepdim=True)
        scale = x.var(dim=1, keepdim=True, unbiased=False).sqrt().clamp_min(1e-5)
        values = ((x - mean) / scale).unsqueeze(-1)
        values = values.to(dtype=next(self.llm.parameters()).dtype)
        output = self.llm(input_ids=values, use_cache=False, return_dict=True)
        # Last causal token has access to the entire input sequence.
        features = output.last_hidden_state[:, -1, :]
        # Restore access to level/scale removed by normalization: the project's
        # distance target is computed from the original input sequences.
        statistics = torch.cat((mean, scale), dim=-1).to(features.dtype)
        return self.projection(torch.cat((features, statistics), dim=-1))
