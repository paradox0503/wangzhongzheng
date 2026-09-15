from utils.conf import Configuration

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, conf):
        super(TokenEmbedding, self).__init__()
        
        self.patch_len = conf.getEntry('patch_len')
        self.dim_model = conf.getEntry('dim_model')
        self.device = conf.getEntry('device')
        
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=self.patch_len, out_channels=self.dim_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
    
    def forward(self, x):
        # x: (batch_size, patch_num, patch_len)
        x = self.tokenConv(x.permute(0, 2, 1)).permute(0, 2, 1)
        # x: (batch_size, patch_num, dim_model)
        return x
        

class Patching(nn.Module):
    def __init__(self, conf: Configuration):
        super(Patching, self).__init__()
        self.patch_len = conf.getEntry('patch_len')
        self.stride = conf.getEntry('stride')
        
        self.token_embedding = TokenEmbedding(conf)

    def forward(self, sample):
        # sample: tensor(batch_size, len_series)
        patches = []
        for ts in sample: 
            patches.append(ts.unfold(dimension=-1, size=self.patch_len, step=self.stride))
        # patches: (batch_size, patch_num, patch_len)
        patches = self.token_embedding(torch.stack(patches))
        # patches: tensor(batch_size, patch_num, dim_model)
        return patches
        # patches: tensor(batch_size, patch_num, dim_model)
        

def calculate_patch_num(conf: Configuration):
    return (conf.getEntry("len_series") - conf.getEntry('patch_len')) // conf.getEntry('stride') + 1