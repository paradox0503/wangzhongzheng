import numpy as np
import torch
from torch import nn
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import logging

from utils.conf import Configuration
from model.prompt import Prompt
from model.mlp import Mlp


class MyLLM4SSS3(nn.Module):
    def __init__(self, conf: Configuration):
        super(MyLLM4SSS3, self).__init__()
        
        self.conf = conf
        self.device = self.conf.getEntry('device')
        self.selected_device = self.conf.getEntry("GPUs")
        self.batch_size = self.conf.getEntry('batch_size')
        self.llm_path = self.conf.getEntry('llm_path')
        self.llm_type = self.conf.getEntry('llm_type')
        self.llm_layers = self.conf.getEntry('llm_layers')
        self.dim_llm = self.conf.getEntry('dim_llm')
        self.len_series = self.conf.getEntry('len_series')
        self.len_reduce = self.conf.getEntry('len_reduce')
        self.llm_pos = self.llm_path + self.llm_type + '/'
        
        self.dim_llm = conf.getEntry('dim_llm')
        self.win_size = conf.getEntry('win_size')
        self.stride = conf.getEntry('stride')
        
        self.encoder_type = conf.getEntry('encoder_type')
        self.decoder_type = conf.getEntry('decoder_type')

        self.mlp_layers = conf.getEntry('mlp_layers')
        self.mlp_dim = conf.getEntry('mlp_dim')
        self.dropout = conf.getEntry('dropout')

        if self.llm_type == 'gpt2':
            self.llm = GPT2Model.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        elif self.llm_type == 'bert':
            self.llm = BertModel.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        elif self.llm_type == 'llama7b':
            self.llm = LlamaModel.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
            
        for _, (name, param) in enumerate(self.llm.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
                logging.info(f'{name}')
            else:
                param.requires_grad = False
            
        self.win_num = (self.len_series - self.win_size) // self.stride + 1
        
        if self.conf.getEntry('activation') == 'relu':
            self.activation = nn.ReLU()
        elif self.conf.getEntry('activation') == 'tanh':
            self.activation = nn.Tanh()
        elif self.conf.getEntry('activation') == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.conf.getEntry('activation') == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        
        if self.encoder_type == 'linear':
            self.inlayer = nn.Linear(self.win_size, self.dim_llm)
        elif self.encoder_type == 'mlp':
            self.inlayer = Mlp(self.win_size, self.dim_llm, self.mlp_dim,self.mlp_layers, self.activation, self.dropout)
        
        if self.decoder_type == 'linear':
            self.flatten = nn.Flatten(start_dim=-2)
            self.outlayer = nn.Sequential(self.flatten, nn.Linear(self.win_num*self.dim_llm, self.len_reduce))
        elif self.decoder_type == 'mlp':
            self.mlp = Mlp(self.dim_llm, self.win_size, self.mlp_dim, self.mlp_layers, self.activation, self.dropout)
            # x: [batch_size, win_num, win_size]
            self.flatten = nn.Flatten(start_dim=-2)
            # x: [batch_size, win_num * win_size = len_series]
            self.projection = nn.Linear(self.win_num * self.win_size, self.len_reduce)
            # x: [batch_size, len_reduce]
            self.outlayer = nn.Sequential(self.mlp, self.flatten, self.projection)
            

    def forward(self, x):
        # x:[batch_size, len_series]
        x = x.unfold(dimension=-1, size=self.win_size, step=self.stride)
        # x:[batch_size, win_num, win_size]
        x = self.inlayer(x.float())
        # x:[batch_size, win_num, dim_llm]
        output_embedding = self.llm(inputs_embeds=x).last_hidden_state
        # output_embedding:[batch_size, win_num, dim_llm]
        x = output_embedding[:, -self.win_num:, :]
        # x:[batch_size, win_num, dim_llm]
        x = self.outlayer(x)
        # x:[batch_size, len_reduce]
        return x