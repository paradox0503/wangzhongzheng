import numpy as np
import torch
from torch import nn
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from einops import rearrange
import pandas as pd
import logging
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv

from utils.conf import Configuration
from model.mlp import Mlp


class MyLLM4SSS10(nn.Module):
    def __init__(self, conf: Configuration):
        super(MyLLM4SSS10, self).__init__()
        
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
        
        self.trend_len = conf.getEntry('trend_len')
        self.seasonal_len = conf.getEntry('seasonal_len')
        self.top_k = conf.getEntry('top_k')
        
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

        self.prompt = Prompt(conf, self.llm.wte.weight)

        self.edge_index_list = []

        even_indices = [i for i in range(self.win_num) if i % 2 == 0]
        odd_indices = [i for i in range(self.win_num) if i % 2 == 1]

        # 偶数结点 → 正下一个奇数结点（i+1）
        for i in even_indices:
            if i + 1 < self.win_num and (i + 1) % 2 == 1:
                self.edge_index_list.append([i, i + 1])

        # 每个奇数结点 → 所有偶数结点
        for j in odd_indices:
            for i in even_indices:
                self.edge_index_list.append([j, i])

        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).T

        self.gat1 = GATConv(self.dim_llm, self.dim_llm, heads=1, dropout=self.dropout).to(self.device)


    def forward(self, x):
        # x:[batch_size, len_series]
        x = x.unfold(dimension=-1, size=self.win_size, step=self.stride)
        # x:[batch_size, win_num, win_size]
        x = self.inlayer(x)
        # x:[batch_size, win_num, dim_llm]
        prompt_embs = self.prompt(x)
        # prompt_embs:[batch_size, win_num, dim_llm]
        B, W, D = x.shape
        x = torch.stack([prompt_embs, x], dim=2).reshape(B, W * 2, D)
        # x:[batch_size, win_num * 2, dim_llm]
        
        def build_batch_graph(x):  # x: [B, N, D]
            B, N, D = x.shape
            W = N // 2

            data_list = []
            for i in range(B):
                data = Data(x=x[i], edge_index=self.edge_index)
                data_list.append(data)
            
            batch = Batch.from_data_list(data_list)
            return batch
        
        current_device = x.device   # save current device
        batch_graph = build_batch_graph(x).to(current_device)

        x = self.gat1(batch_graph.x, batch_graph.edge_index)
        # x:[batch_size * win_num * 2, dim_llm]
        x = x.view(-1, 2 * self.win_num, self.dim_llm)
        # x:[batch_size, win_num * 2, dim_llm]
        output_embedding = self.llm(inputs_embeds=x).last_hidden_state
        # output_embedding:[batch_size, win_num * 2, dim_llm]
        x = output_embedding[:, 1::2, :]
        # x:[batch_size, win_num, dim_llm]
        x = self.outlayer(x)
        # x:[batch_size, len_reduce]
        return x


class Prompt(nn.Module):
    def __init__(self, conf:Configuration, wte):
        super(Prompt, self).__init__()

        self.conf = conf
        self.wte = wte
        self.pool_size = conf.getEntry('pool_size')
        self.dim_llm = conf.getEntry('dim_llm')

        self.text_prototype_linear = nn.Linear(wte.shape[0], self.pool_size)


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    

    def forward(self, x):
        # x:[batch_size, win_num, dim_llm]
        win_num = x.shape[1]
        
        prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)
        # prompt_key:[pool_size, dim_llm]

        x_norm = self.l2_normalize(x, dim=2)
        # x_norm:[batch_size, win_num, dim_llm]
        prompt_key_norm = self.l2_normalize(prompt_key, dim=1)
        # prompt_key_norm:[pool_size, dim_llm]

        similarity = torch.matmul(x_norm, prompt_key_norm.transpose(0, 1))
        # sim:[batch_size, win_num, pool_size]

        weights = torch.softmax(similarity, dim=2)
        # weights:[batch_size, win_num, pool_size]

        prompt = torch.matmul(weights, prompt_key_norm)
        # [batch_size, win_num, pool_size] * [pool_size, dim_llm] ——> [batch_size, win_num, dim_llm]

        return prompt
