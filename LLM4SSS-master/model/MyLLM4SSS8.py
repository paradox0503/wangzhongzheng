from collections import Counter
import numpy as np
from regex import F
import torch
from torch import nn
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import logging
import math


from utils.conf import Configuration


class MyLLM4SSS8(nn.Module):
    def __init__(self, conf: Configuration):
        super(MyLLM4SSS8, self).__init__()
        
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

        self.patch_size = conf.getEntry('patch_size')
        self.stride = conf.getEntry('stride')
        self.patch_num = (self.len_series - self.patch_size) // self.stride + 1
        
        self.dropout = conf.getEntry('dropout')
        
        self.k = conf.getEntry('k')
        
        if self.llm_type == 'gpt2':
            config = GPT2Config.from_pretrained(self.llm_pos)
            config.output_hidden_states = False
            self.llm = GPT2Model.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        elif self.llm_type == 'bert':
            config = GPT2Config.from_pretrained(self.llm_pos)
            config.output_hidden_states = False
            self.llm = BertModel.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        elif self.llm_type == 'llama7b':
            config = GPT2Config.from_pretrained(self.llm_pos)
            config.output_hidden_states = False
            self.llm = LlamaModel.from_pretrained(self.llm_pos).to(self.device)  # pyright: ignore
        else:
            raise ValueError("Invalid model selected.")
            
        for _, (name, param) in enumerate(self.llm.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
                logging.info(f'{name}')
            else:
                param.requires_grad = False
        
        if self.conf.getEntry('activation') == 'relu':
            self.activation = nn.ReLU()
        elif self.conf.getEntry('activation') == 'tanh':
            self.activation = nn.Tanh()
        elif self.conf.getEntry('activation') == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.conf.getEntry('activation') == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Invalid activation selected.")

        self.conv2d_row = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2d_column = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.inlayer = nn.Linear(self.patch_size, self.dim_llm)
        
        self.position_embedding = PositionalEmbedding(self.dim_llm)
        
        self.outlayer = nn.Linear(self.dim_llm, self.patch_size)
        
        self.flatten = nn.Flatten(-2)
        
        self.row_weight = nn.Parameter(torch.ones(1))
        self.column_weight = nn.Parameter(torch.ones(1))
        
        self.projection = nn.Linear(self.len_series, self.len_reduce)
        
        self.weights = nn.Parameter(torch.ones(self.k))
            

    def forward(self, x):
        # x:[batch_size, len_series]

        top_k_periods, _, _ = find_top_k_periods_batch(x.cpu().numpy(), self.conf)
        # top_k_periods:[batch_size, k]
        top_k_periods = top_k_periods.reshape(-1)
        # top_k_periods:[batch_size * k]
        period_counts = Counter(top_k_periods)
        k_periods = period_counts.most_common(self.k)
        # most_common_periods:[(period, count), ...]
        k_periods = [period[0] for period in k_periods]
        # k_periods:[period1, period2, ...]
        k_season_size = k_periods
        # k_win_size:[period1, period2, ...]

        result = torch.zeros(x.shape[0], self.len_reduce, device=self.device, dtype=x.dtype)
        # result:[batch_size, len_reduce]

        for season_size, weight in zip(k_season_size, self.weights):

            if self.len_series % season_size != 0:
                pad_num = season_size - (self.len_series % season_size)
                padding_method = 'zero'
                x_cat = torch.cat([x, torch.zeros((x.shape[0], pad_num)).to(self.device)], dim=-1)
                # x:[batch_size, len_series + pad_num]
            else:
                x_cat = x
            
            row = x_cat.unfold(dimension=-1, size=season_size, step=season_size)
            # row:[batch_size, season_num, season_size]
            column = row.transpose(1, 2)
            # column:[batch_size, season_size, season_num]
            row, column = row.unsqueeze(1), column.unsqueeze(1)
            # row:[batch_size, 1, season_num, season_size]   column:[batch_size, 1, season_size, season_num]
            row, column = self.conv2d_row(row), self.conv2d_column(column)
            # row:[batch_size, 1, season_num, season_size]   column:[batch_size, 1, season_size, season_num]
            row, column = row.squeeze(1), column.squeeze(1)
            # row:[batch_size, season_num, season_size]   column:[batch_size, season_size, season_num]
            row, column = self.flatten(row), self.flatten(column)
            # row:[batch_size, len_series + pad_num]   column:[batch_size, len_series + pad_num]
            row, column = row[:, :self.len_series], column[:, :self.len_series]
            # row:[batch_size, len_series]   column:[batch_size, len_series]
            row, column = row.unfold(dimension=-1, size=self.patch_size, step=self.stride), column.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            # row:[batch_size, patch_num, patch_size]   column:[batch_size, patch_num, patch_size]
            x_row, x_column = self.inlayer(row), self.inlayer(column)
            # x_row:[batch_size, patch_num, dim_llm]   x_column:[batch_size, patch_num, dim_llm]
            x_row, x_column = x_row + self.position_embedding(x_row), x_column + self.position_embedding(x_column)
            # x_row:[batch_size, patch_num, dim_llm]   x_column:[batch_size, patch_num, dim_llm]
            x_cat = torch.cat([x_row, x_column], dim=1)
            # x_cat:[batch_size, patch_num + patch_num, dim_llm]
            with torch.cuda.amp.autocast():
                out = self.llm(inputs_embeds=x_cat).last_hidden_state
            # out:[batch_size, patch_num + patch_num, dim_llm]
            out_row, out_column = out[:, :self.patch_num, :], out[:, -self.patch_num:, :]
            # out_row:[batch_size, patch_num, dim_llm]   out_column:[batch_size, patch_num, dim_llm]
            out_row, out_column = self.outlayer(out_row), self.outlayer(out_column)
            # out_row:[batch_size, patch_num, patch_size]   out_column:[batch_size, patch_num, patch_size]
            out_row, out_column = self.flatten(out_row), self.flatten(out_column)
            # out_row:[batch_size, len_series]   out_column:[batch_size, len_series]
            x_result = out_row * self.row_weight + out_column * self.column_weight
            # x_result:[batch_size, len_series]
            x_project = self.projection(x_result)
            # x_project:[batch_size, len_reduce]
            
            result += weight * x_project
        # result:[batch_size, len_reduce]
        return result
    

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
        # x_row:[batch_size, patch_num, dim_llm]
        # x_column:[batch_size, win_size, dim_llm]
        return self.pe[:, :x.size(1), :]
    
    
def find_top_k_periods_batch(series_batch, conf: Configuration):
    # series_batch: [batch_size, len_series]
    k = conf.getEntry("k")
    batch_size, len_series = series_batch.shape
    max_period = conf.getEntry("max_win_size")
    min_period = conf.getEntry("min_win_size")
    
    fft_coeffs = np.fft.fft(series_batch, axis=1)
    amplitudes = np.abs(fft_coeffs)

    # calculate frequencies
    freqs = np.fft.fftfreq(len_series) * len_series
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    amplitudes = amplitudes[:, positive_mask]

    # calculate periods
    periods = np.ceil(len_series / freqs).astype(int)

    # filter periods greater than max_period
    valid_mask = (periods <= max_period) & (periods >= min_period)
    freqs = freqs[valid_mask]
    amplitudes = amplitudes[:, valid_mask]
    periods = periods[valid_mask]

    # sort to select top k periods for each sequence
    sorted_indices = np.argsort(amplitudes, axis=1)[:, ::-1]

    top_k_periods = periods[sorted_indices[:, :k]]
    top_k_freqs = freqs[sorted_indices[:, :k]]
    top_k_amplitudes = amplitudes[np.arange(batch_size)[:, None], sorted_indices[:, :k]]

    return top_k_periods, top_k_freqs, top_k_amplitudes