import numpy as np
import torch
from torch import nn
from transformers import LlamaModel, LlamaTokenizer, LlamaConfig
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import logging

from utils.conf import Configuration


class MyLLM4SSS9(nn.Module):
    def __init__(self, conf: Configuration):
        super(MyLLM4SSS9, self).__init__()

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

        self.win_num = (self.len_series - self.win_size) // self.stride + 1

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
        
        self.inlayer = nn.Linear(self.win_size, self.dim_llm)
        self.flatten = nn.Flatten(-2)
        self.outlayer = nn.Sequential(self.flatten, nn.Linear(self.win_num*self.dim_llm, self.len_reduce))

    
    def forward(self, x1, x2):
        # x1, x2:[batch_size, len_series]
        x1 = x1.unfold(dimension=-1, size=self.win_size, step=self.stride)
        x2 = x2.unfold(dimension=-1, size=self.win_size, step=self.stride)
        # x1, x2:[batch_size, win_num, win_size]
        x1 = self.inlayer(x1)
        x2 = self.inlayer(x2)
        # x1, x2:[batch_size, win_num, dim_llm]
        in_embedding = torch.cat([x1, x2], dim=1)
        # in_embedding:[batch_size, 2*win_num, dim_llm]
        with torch.cuda.amp.autocast():
            out_embedding = self.llm(inputs_embeds=in_embedding).last_hidden_state
        # out_embedding:[batch_size, 2*win_num, dim_llm]
        x1 = out_embedding[:, :self.win_num, :]
        x2 = out_embedding[:, -self.win_num:, :]
        # x1, x2:[batch_size, win_num, dim_llm]
        x1 = self.outlayer(x1)
        x2 = self.outlayer(x2)
        # x1, x2:[batch_size, len_reduce]
        return x1, x2
