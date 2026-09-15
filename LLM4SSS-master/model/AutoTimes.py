import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
import logging

from utils import conf
from utils.conf import Configuration
from model.mlp import Mlp


class AutoTimes(nn.Module):
    def __init__(self, conf:Configuration):
        super(AutoTimes, self).__init__()
        
        self.conf = conf
        self.device = self.conf.getEntry('device')
        self.batch_size = self.conf.getEntry('batch_size')
        self.llm_path = self.conf.getEntry('llm_path')
        self.llm_type = self.conf.getEntry('llm_type')
        self.llm_layers = self.conf.getEntry('llm_layers')
        self.dim_llm = self.conf.getEntry('dim_llm')
        self.len_series = self.conf.getEntry('len_series')
        self.len_reduce = self.conf.getEntry('len_reduce')
        self.llm_pos = self.llm_path + self.llm_type + '/'

        self.win_size = conf.getEntry('win_size')
        self.stride = conf.getEntry('stride')
        self.mlp_layers = conf.getEntry('mlp_layers')
        self.mlp_dim = conf.getEntry('mlp_dim')
        self.dropout = conf.getEntry('dropout')
        
        if self.llm_type == 'gpt2':
            self.llm = GPT2Model.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        elif self.llm_type == 'bert':
            self.llm = BertModel.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        elif self.llm_type == 'llama7b':
            self.llm = llamaModel.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore

        for name, param in self.llm.named_parameters():
            param.requires_grad = False

        if self.conf.getEntry('activation') == 'relu':
            self.activation = nn.ReLU()
        elif self.conf.getEntry('activation') == 'tanh':
            self.activation = nn.Tanh()
        elif self.conf.getEntry('activation') == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.conf.getEntry('activation') == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.encoder = Mlp(self.win_size, self.dim_llm, self.mlp_dim,
                           self.mlp_layers, self.activation, self.dropout)
        self.decoder = Mlp(self.dim_llm, self.win_size, self.mlp_dim,
                           self.mlp_layers, self.activation, self.dropout)

        self.projection = nn.Linear(self.len_series, self.len_reduce)
        
        
    def forward(self, x):
        # x:[batch_size, len_series]
        x_enc = x.unfold(dimension=-1, size=self.win_size, step=self.stride)
        # x_enc:[batch_size, token_num, win_size]
        embedding = self.encoder(x_enc)
        # embedding:[batch_size, token_num, dim_llm]
        output_embedding = self.llm(inputs_embeds=embedding).last_hidden_state
        # output_embedding:[batch_size, token_num, dim_llm]
        x_dec = self.decoder(output_embedding)
        # x_dec:[batch_size, token_num, win_size]
        x_dec = x_dec.reshape(-1, self.len_series)
        # x_dec:[batch_size, len_series]
        x_dec = self.projection(x_dec)
        # x_dec:[batch_size, len_reduce]
        return x_dec