"""AutoTimes-style patch adapter around the pretrained TimeMoE-50M decoder."""

import torch
from torch import nn
from pathlib import Path

from utils.conf import Configuration
from model.mlp import Mlp


class TimeMoE(nn.Module):
    def __init__(self, conf: Configuration):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM

        self.len_series = conf.getEntry('len_series')
        self.len_reduce = conf.getEntry('len_reduce')
        self.win_size = conf.getEntry('win_size')
        self.stride = conf.getEntry('stride')
        if (self.win_size <= 0 or self.stride != self.win_size
                or self.len_series < self.win_size
                or self.len_series % self.win_size != 0):
            raise ValueError('TimeMoE requires win_size == stride > 0 and '
                             'len_series divisible by win_size (no dropped tail).')
        self.patch_num = self.len_series // self.win_size
        self.freeze_backbone = conf.getEntry('freeze_backbone')
        source = conf.getEntry('time_moe_path')
        if source.startswith('~'):
            source = str(Path(source).expanduser())
        local_only = conf.getEntry('local_files_only')
        config = AutoConfig.from_pretrained(source, trust_remote_code=True,
                                            local_files_only=local_only)
        expected = dict(model_type='time_moe', hidden_size=384, num_hidden_layers=12,
                        num_experts=8, input_size=1, use_dense=False)
        if any(getattr(config, key, None) != value for key, value in expected.items()):
            raise ValueError('TimeMoE requires the Maple728/TimeMoE-50M base checkpoint.')
        if self.patch_num > config.max_position_embeddings:
            raise ValueError('Patch count exceeds the TimeMoE context length.')
        # Official TimeMoE implements eager and FlashAttention2, not SDPA.
        attention = getattr(conf, 'confLoaded', {}).get('attn_implementation', 'eager')
        if attention not in ('eager', 'flash_attention_2'):
            raise ValueError('TimeMoE attn_implementation must be eager or flash_attention_2.')
        pretrained = AutoModelForCausalLM.from_pretrained(
            source, config=config, trust_remote_code=True, local_files_only=local_only,
            torch_dtype=torch.float32, attn_implementation=attention,
        )
        # Use hidden features, not forecast logits or autoregressive generation.
        # Naming matches the existing initializer's pretrained-module exclusion.
        self.llm = pretrained.model
        self.llm.config.use_cache = False
        self.llm.requires_grad_(not self.freeze_backbone)
        if self.freeze_backbone:
            self.llm.eval()
        activations = dict(relu=nn.ReLU, tanh=nn.Tanh, sigmoid=nn.Sigmoid,
                           leakyrelu=nn.LeakyReLU)
        activation_name = conf.getEntry('activation')
        if activation_name not in activations:
            raise ValueError(f'Unsupported activation: {activation_name}')
        activation = activations[activation_name]()
        self.encoder = Mlp(self.win_size, config.hidden_size, conf.getEntry('mlp_dim'),
                           conf.getEntry('mlp_layers'), activation, conf.getEntry('dropout'))
        self.decoder = Mlp(config.hidden_size, self.win_size, conf.getEntry('mlp_dim'),
                           conf.getEntry('mlp_layers'), activation, conf.getEntry('dropout'))
        self.projection = nn.Linear(self.len_series, self.len_reduce)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.llm.eval()
        return self

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.len_series:
            raise ValueError('TimeMoE expects [batch_size, len_series].')
        # Match AutoTimes: raw sequence -> non-overlapping patches -> MLP tokens.
        patches = x.unfold(dimension=-1, size=self.win_size, step=self.stride)
        embedding = self.encoder(patches)
        embedding = embedding.to(dtype=next(self.llm.parameters()).dtype)
        # Do not use no_grad here: the trainable encoder needs gradients through
        # the frozen decoder. Exporters already use inference_mode externally.
        output = self.llm(inputs_embeds=embedding, use_cache=False, return_dict=True,
                          output_attentions=False, output_hidden_states=False)
        decoded = self.decoder(output.last_hidden_state)
        return self.projection(decoded.reshape(x.shape[0], self.len_series))
