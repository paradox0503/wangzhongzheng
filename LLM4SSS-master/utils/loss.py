import torch
from torch import nn
import numpy as np
from numpy import sqrt
from torch.nn.modules.distance import PairwiseDistance
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class ScaledL2Loss(nn.Module):
    def __init__(self, len_series: int, len_reduce: int):
        super(ScaledL2Loss, self).__init__()

        self.l2 = PairwiseDistance(p=2).cuda()
        self.l1 = PairwiseDistance(p=1).cuda()
        self.scale_factor_original = sqrt(float(len_series))
        self.scale_factor_reduce = sqrt(float(len_reduce))


    def forward(self, one, another, one_reduce, another_reduce):
        # one: (batch_size, len_series), another: (batch_size, len_reduce)
        
        original_l2 = self.l2(one, another) / self.scale_factor_original    # (batch_size)
        reduce_l2 = self.l2(one_reduce, another_reduce) / self.scale_factor_reduce    # (batch_size)

        return self.l1(original_l2.reshape(1, -1), reduce_l2.reshape(1, -1))[0] / one.shape[0]    
        # (1, batch_size) -> scalar


class ExpScaledL2Loss(nn.Module):
    def __init__(self, len_series: int, len_reduce: int):
        super(ExpScaledL2Loss, self).__init__()
        
        self.l2 = PairwiseDistance(p=2).cuda()
        self.l1 = PairwiseDistance(p=1).cuda()
        self.scale_factor_original = sqrt(float(len_series))
        self.scale_factor_reduce = sqrt(float(len_reduce))
    
    
    def forward(self, one, another, one_reduce, another_reduce):
        # one: (batch_size, len_series), another: (batch_size, len_reduce)

        original_l2 = self.l2(one, another) / self.scale_factor_original    # (batch_size)
        reduce_l2 = self.l2(one_reduce, another_reduce) / self.scale_factor_reduce    # (batch_size)

        diff = torch.abs(original_l2 - reduce_l2)    # (batch_size)
        exp_diff = torch.exp(diff) - 1   # (batch_size)

        return torch.mean(exp_diff)
    
    
class LogScaledL2Loss(nn.Module):
    def __init__(self, len_series: int, len_reduce: int):
        super(LogScaledL2Loss, self).__init__()

        self.l2 = PairwiseDistance(p=2).cuda()
        self.scale_factor_original = sqrt(float(len_series))
        self.scale_factor_reduce = sqrt(float(len_reduce))

    def forward(self, one, another, one_reduce, another_reduce):
        # one, another: (batch_size, len_series)

        original_l2 = self.l2(one, another) / self.scale_factor_original  # (batch_size)
        reduce_l2 = self.l2(one_reduce, another_reduce) / self.scale_factor_reduce  # (batch_size)

        diff = torch.abs(original_l2 - reduce_l2)  # (batch_size)
        log_diff = torch.log1p(diff)  # log(1 + diff), (batch_size)

        return torch.mean(log_diff)


class Umap(nn.Module):
    def __init__(self, len_series: int, len_reduce: int, n_neighbors=15):
        super(Umap, self).__init__()
        self.len_series = len_series
        self.len_reduce = len_reduce
        self.n_neighbors = n_neighbors
        
        # 预计算 NearestNeighbors 和 sigmas, rhos
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.sigmas = None
        self.rhos = None
    
    def compute_high_dim_probs(self, X):
        X_np = X.detach().cpu().numpy()
        n = X_np.shape[0]

        self.knn.fit(X_np)
        knn_dists, knn_inds = self.knn.kneighbors(X_np, return_distance=True)

        sigmas = np.zeros(n)
        rhos = np.zeros(n)
        for i in range(n):
            rho = knn_dists[i, 1]
            rhos[i] = rho
            lo, hi = 0.0, np.inf
            mid = 1.0
            target = np.log2(self.n_neighbors)
            for _ in range(64):
                psum = 0.0
                for d in knn_dists[i, 1:]:
                    psum += np.exp(-(max(0.0, d - rho)) / mid)
                if abs(psum - target) < 1e-5:
                    break
                if psum > target:
                    hi = mid
                    mid = (lo + hi) / 2 if hi < np.inf else mid / 2
                else:
                    lo = mid
                    mid = (hi + lo) / 2 if hi < np.inf else mid * 2
            sigmas[i] = mid

        self.sigmas = sigmas
        self.rhos = rhos

        P = np.zeros((n, n))
        for i in range(n):
            for j in range(1, self.n_neighbors):
                j_idx = knn_inds[i, j]
                d = knn_dists[i, j]
                p = np.exp(-(max(0.0, d - rhos[i])) / sigmas[i])
                P[i, j_idx] = p

        P = P + P.T - P * P.T
        P = torch.tensor(P, dtype=torch.float32, device=X.device)
        return P  # [N, N]


    def compute_similarity(self, reduce):
        dist_sq = torch.cdist(reduce, reduce, p=2) ** 2  # [B, B]
        Q = 1.0 / (1.0 + 1.929 * dist_sq ** 0.7915)
        Q = torch.clamp(Q, min=1e-4, max=1 - 1e-4)
        return Q


    def forward(self, origin, reduce):
        # reduce: [batch_size, dim]
        P = self.compute_high_dim_probs(origin)  # [batch_size, batch_size]
        Q = self.compute_similarity(reduce)  # [batch_size, batch_size]

        BCE = P * torch.log(Q) + (1 - P) * torch.log(1 - Q)
        loss = -torch.mean(BCE)
        return loss
