import numpy as np
import torch
from torch import nn
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import logging
import math


from utils.conf import Configuration


class MyLLM4SSS6(nn.Module):
    def __init__(self, conf: Configuration):
        super(MyLLM4SSS6, self).__init__()
        
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
            
        self.win_num = (self.len_series - self.win_size) // self.win_size + 1
        
        if self.conf.getEntry('activation') == 'relu':
            self.activation = nn.ReLU()
        elif self.conf.getEntry('activation') == 'tanh':
            self.activation = nn.Tanh()
        elif self.conf.getEntry('activation') == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.conf.getEntry('activation') == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.inlayer_row = nn.Linear(self.win_size, self.dim_llm)
        self.inlayer_column = nn.Linear(self.win_num, self.dim_llm)
        
        self.position_embedding = PositionalEmbedding(self.dim_llm)
        
        self.outlayer_row = nn.Linear(self.dim_llm, self.win_size)
        self.outlayer_column = nn.Linear(self.dim_llm, self.win_num)
        
        self.flatten = nn.Flatten(-2)
        
        self.row_weight = nn.Parameter(torch.ones(1))
        self.column_weight = nn.Parameter(torch.ones(1))
        
        self.projection = nn.Linear(self.len_series, self.len_reduce)
            

    def forward(self, x):
        # x:[batch_size, len_series]
        row = x.unfold(dimension=-1, size=self.win_size, step=self.win_size)
        # row:[batch_size, win_num, win_size]
        column = row.transpose(1, 2)
        # column:[batch_size, win_size, win_num]
        x_row = self.inlayer_row(row.float())
        # x_row:[batch_size, win_num, dim_llm]
        x_column = self.inlayer_column(column.float())
        # x_column:[batch_size, win_size, dim_llm]
        x_row = x_row + self.position_embedding(x_row)
        # x_row:[batch_size, win_num, dim_llm]
        x_column = x_column + self.position_embedding(x_column)
        # x_column:[batch_size, win_size, dim_llm]
        x = torch.cat([x_row, x_column], dim=1)
        # x:[batch_size, win_num + win_size, dim_llm]
        out = self.llm(inputs_embeds=x).last_hidden_state
        # output_embedding:[batch_size, win_num + win_size, dim_llm]
        out_row = out[:, :self.win_num, :]
        # out_row:[batch_size, win_num, dim_llm]
        out_column = out[:, -self.win_size:, :]
        # out_column:[batch_size, win_size, dim_llm]
        out_row = self.outlayer_row(out_row)
        # out_row:[batch_size, win_num, win_size]
        out_column = self.outlayer_column(out_column)
        # out_column:[batch_size, win_size, win_num]
        out_row = self.flatten(out_row)
        # out_row:[batch_size, len_series]
        out_column = self.flatten(out_column)
        # out_column:[batch_size, len_series]
        x = out_row * self.row_weight + out_column * self.column_weight
        # x:[batch_size, len_series]
        x = self.projection(x)
        # x:[batch_size, len_reduce]
        return x
    

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x_row:[batch_size, win_num, dim_llm]
        # x_column:[batch_size, win_size, dim_llm]
        return self.pe[:, :x.size(1), :]