import torch
from torch import nn

from utils.conf import Configuration


class Prompt(nn.Module):
    def __init__(self, conf:Configuration, wte):
        super(Prompt, self).__init__()
        
        self.conf = conf
        self.dim_llm = conf.getEntry('dim_llm')
        
        self.pool_size = conf.getEntry('pool_size')
        self.prompt_key_init = conf.getEntry('prompt_key_init')
        self.embedding_key = conf.getEntry('embedding_key')
        self.top_k = conf.getEntry('top_k')
        self.wte = wte
        
        prompt_key_shape = (self.pool_size, self.dim_llm)
        if self.prompt_key_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(prompt_key_shape),requires_grad=False)
        elif self.prompt_key_init == 'uniform':
            self.prompt = nn.Parameter(torch.randn(prompt_key_shape),requires_grad=False)
            nn.init.uniform_(self.prompt, -5, 5)
        elif self.prompt_key_init == 'gaussian':
            self.prompt = nn.Parameter(torch.randn(prompt_key_shape),requires_grad=False)
            nn.init.normal_(self.prompt, mean=0.0, std=5.0)
        elif self.prompt_key_init == 'text_prototype':
            self.text_prototype_linear = nn.Linear(wte.shape[0], self.pool_size)
        # prompt:[pool_size, dim_llm]

        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm


    def forward(self, x):
        # x:[batch_size, win_num, dim_llm]
        outs = dict()
        
        if self.embedding_key == 'mean':
            x_repre = torch.mean(x, dim=1)
        elif self.embedding_key == 'max':
            x_repre = torch.max(x, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_repre = torch.max(x, dim=1)[0] + 2 * torch.mean(x, dim=1)
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
        # x_repre:[batch_size, dim_llm]
        
        if self.prompt_key_init == 'text_prototype':
            # wte:[vocab_size, dim_llm]
            prompt_key = self.text_prototype_linear(self.wte.transpose(0, 1)).transpose(0, 1)   #pyright:ignore
            # prompt_key:[pool_size, dim_llm]
        else:
            prompt_key = self.prompt
            # prompt:[pool_size, dim_llm]
        
        prompt_key_norm = self.l2_normalize(prompt_key, dim=1)
        # prompt_key_norm:[pool_size, dim_llm]
        
        x_repre_norm = self.l2_normalize(x_repre, dim=1)
        # x_repre_norm:[batch_size, dim_llm]
        
        similarity = torch.matmul(x_repre_norm, prompt_key_norm.t())
        # x_repre_norm:[batch_size, dim_llm] * prompt_key_norm.t(): [dim_llm, pool_size]
        # similarity:[batch_size, pool_size]
        
        _, index = torch.topk(similarity, k=self.top_k, dim=1)
        # index:[batch_size, top_k]
        
        prompt = prompt_key[index]
        # prompt:[batch_size, top_k, dim_llm]
        
        outs['prompt_index'] = index
        outs['prompted_embedding'] = torch.cat([prompt, x], dim=1)
        # prompted_embedding:[batch_size, top_k + win_num, dim_llm]
        
        return outs