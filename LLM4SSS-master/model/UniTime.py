import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import BertModel, BertTokenizer, BertConfig
import math
import logging

from utils.conf import Configuration
from model.unitimegpt2 import UniTimeGPT2


class FlattenHead(nn.Module):
    def __init__(self, dim_in, dim_out, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(dim_in, dim_out)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class UniTime(nn.Module):
    def __init__(self, conf: Configuration):
        super(UniTime, self).__init__()

        self.conf = conf
        self.device = self.conf.getEntry('device')
        self.batch_size = self.conf.getEntry('batch_size')
        self.llm_path = self.conf.getEntry('llm_path')
        self.llm_type = self.conf.getEntry('llm_type')
        self.llm_layers = self.conf.getEntry('llm_layers')
        self.dim_llm = self.conf.getEntry('dim_llm')
        self.len_series = self.conf.getEntry('len_series')
        self.len_reduce = self.conf.getEntry('len_reduce')
        self.patch_len = self.conf.getEntry('patch_len')
        self.stride = self.conf.getEntry('stride')
        self.mask_rate = self.conf.getEntry('mask_rate')
        self.dropout = self.conf.getEntry('dropout')
        self.dec_layernum = self.conf.getEntry('dec_layernum')

        self.llm_pos = self.llm_path + self.llm_type + '/'

        self.backbone = GPT2Model.from_pretrained(self.llm_pos)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(self.llm_pos)

        # self.backbone.transformer.h = self.backbone.transformer.h[:self.llm_layers]   # pyright:ignore
        
        # freezing
        logging.info('Layer Unfrozen : ')
        for _, (name, param) in enumerate(self.backbone.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
                logging.info(f'{name}')
            else:
                param.requires_grad = False

        config = self.backbone.config   # pyright:ignore

        self.feature_embedding = nn.Linear(self.patch_len, self.dim_llm)

        if self.mask_rate > 0:
            self.feature_projection = nn.Linear(self.dim_llm, self.dim_llm)
            self.binary_indicator_embedding = nn.Linear(self.patch_len, self.dim_llm)
            self.gate_w1 = nn.Linear(self.dim_llm, self.dim_llm)
            self.gate_w2 = nn.Linear(self.dim_llm, self.dim_llm)
            self.gate_sigmoid = nn.Sigmoid()

        self.ts_embed_dropout = nn.Dropout(self.dropout)

        dec_trans_layer = nn.TransformerEncoderLayer(d_model=self.dim_llm,
                                                     nhead=config.n_head,
                                                     dim_feedforward=self.dim_llm * 4, 
                                                     dropout=config.attn_pdrop,
                                                     layer_norm_eps=config.layer_norm_epsilon,
                                                     batch_first=True,
                                                     norm_first=False)
        self.dec_transformer = nn.TransformerEncoder(dec_trans_layer, num_layers=self.dec_layernum)

        patch_num = (self.len_series - self.patch_len) // self.stride + 1
        self.dec_head = FlattenHead(self.dim_llm * patch_num, self.len_reduce, self.dropout)

    def generate_token(self, x, mask):
        # x/mask:  (batch_size, len_series)
        # print("2:",x.shape)
        
        self.stride = self.conf.getEntry('stride')
        
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        mask = mask.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x/mask:  (batch_size, patch_num, patch_len)
        # print("3:",x.shape)
        
        self.patch_num = x.shape[1]

        x_embed = self.feature_embedding(x)
        # x_embed: (batch_size, patch_num, dim_llm)

        if self.mask_rate > 0:
            mask_embed = self.binary_indicator_embedding(mask)
            # mask_embed: (batch_size, patch_num, dim_llm)
            gate = self.gate_sigmoid(self.gate_w1(x_embed) + self.gate_w2(mask_embed))
            x_embed = gate * x_embed + (1 - gate) * mask_embed
            x_embed = self.feature_projection(x_embed)

        return self.ts_embed_dropout(x_embed)

    
    def forward(self, x_union):
        # x/mask: (batch_size, len_series)
        # print("1:",x.shape)
        x, mask = x_union
        # mask = self.random_mask()
        x_embed = self.generate_token(x, mask)
        # x_embed: (batch_size, patch_num, dim_llm)
        # print("4:",x_embed.shape)

        x_enc = self.backbone(inputs_embeds=x_embed).last_hidden_state   # pyright:ignore
        # x_enc: (batch_size, patch_num, dim_llm)
        # print("5:",x_enc.shape)

        x_dec = self.dec_transformer(x_enc)
        # x_dec: (batch_size, patch_num, dim_llm)
        # print("6:",x_dec.shape)

        x_out = self.dec_head(x_dec)
        # x_out: (batch_size, len_reduce)
        # print("7:",x_out.shape)

        return x_out