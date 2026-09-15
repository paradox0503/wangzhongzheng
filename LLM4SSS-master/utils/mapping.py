import torch
from torch import nn

from utils.conf import Configuration


class Mapping(nn.Module):
    def __init__(self, conf: Configuration, attention_dropout=0.1):
        super(Mapping, self).__init__()
        
        self.dim_model = conf.getEntry('dim_model')
        self.dim_llm = conf.getEntry('dim_llm')
        self.head_num = conf.getEntry('head_num')
        self.device = conf.getEntry('device')
        
        self.dim_keys = self.dim_model // self.head_num
        
        self.query_projection = nn.Linear(self.dim_model, self.dim_keys * self.head_num)
        self.key_projection = nn.Linear(self.dim_llm, self.dim_keys * self.head_num)
        self.value_projection = nn.Linear(self.dim_llm, self.dim_keys * self.head_num)
        self.out_projection = nn.Linear(self.dim_keys * self.head_num, self.dim_llm)
        
        self.dropout = nn.Dropout(attention_dropout)
        
        
    def forward(self, query, key, value):
        # query: (batch_size, patch_num, dim_model), key/value: (reduced_vocab_size, dim_llm)
        
        self.batch_size = query.size(0)
        self.patch_num = query.size(1)
        self.reduced_vocab_size = key.size(0)
        
        query = self.query_projection(query).view(self.batch_size, self.patch_num, self.head_num, self.dim_keys)
        # query: (batch_size, patch_num, dim_model) -> (batch_size, patch_num, head_num * dim_keys) -> (batch_size, patch_num, head_num, dim_keys)
        key = self.key_projection(key).view(self.reduced_vocab_size, self.head_num, self.dim_keys)
        # key: (reduced_vocab_size, dim_llm) -> (reduced_vocab_size, head_num * dim_keys) -> (reduced_vocab_size, head_num, dim_keys)
        value = self.value_projection(value).view(self.reduced_vocab_size, self.head_num, self.dim_keys)
        # value: (reduced_vocab_size, dim_llm) -> (reduced_vocab_size, head_num * dim_keys) -> (reduced_vocab_size, head_num, dim_keys)
        
        scale = 1. / self.dim_keys
        
         # (batch_size, patch_num, head_num, dim_keys) * (reduced_vocab_size, head_num, dim_keys)
        scores = torch.einsum("bphk,vhk->bhpv", query, key)
        # score: (batch_size, head_num, patch_num, reduced_vocab_size)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # A: (batch_size, head_num, patch_num, reduced_vocab_size)
        
        # (batch_size, head_num, patch_num, reduced_vocab_size) * (reduced_vocab_size, head_num, dim_keys)
        batch_mapped = torch.einsum("bhpv,vhk->bphk", A, value)
        # batch_mapped: (batch_size, patch_num, head_num, dim_keys)
        
        batch_mapped = batch_mapped.contiguous().view(self.batch_size, self.patch_num, self.head_num * self.dim_keys)
        # batch_mapped: (batch_size, patch_num, head_num * dim_keys)
        
        batch_mapped = self.out_projection(batch_mapped)
        # batch_mapped: (batch_size, patch_num, dim_llm)
        
        return batch_mapped
        # batch_mapped: (batch_size, patch_num, dim_llm)