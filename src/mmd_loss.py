"""
Calculates the Maximum Mean Discrepancy (MMD) loss between two feature distributions.
This module is used to mathematically align the trainable SAR feature space with the
frozen EO reference during the model's fine-tuning.
"""
# Obtained from https://github.com/yiftachbeer/mmd_loss_pytorch

import torch
import torch.nn as nn

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.register_buffer('bandwidth_multipliers', mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        bw = (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]
        return torch.exp(-L2_distances[None, ...] / bw).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY