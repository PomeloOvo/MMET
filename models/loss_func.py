"""Loss functions used by MetaPatchET training."""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde


class DensityWeightedMSE(nn.Module):
    """Weight squared errors inversely by training-label density."""

    def __init__(self, all_train_labels_norm, max_weight=3.0):
        super().__init__()
        self.max_weight = max_weight
        self.kde = gaussian_kde(all_train_labels_norm)

        densities = self.kde(all_train_labels_norm)
        weights = 1.0 / (densities + 1e-6)
        self.mean_weight = np.mean(weights)
        print(f"DensityWeightedMSE initialized (KDE bandwidth={self.kde.factor:.4f})")

    def forward(self, pred, target):
        target_cpu = target.detach().cpu().numpy().flatten()
        batch_densities = self.kde(target_cpu)
        batch_weights = 1.0 / (batch_densities + 1e-6)
        batch_weights = batch_weights / self.mean_weight
        batch_weights = torch.tensor(
            batch_weights,
            dtype=torch.float32,
            device=pred.device
        )
        batch_weights = torch.clamp(batch_weights, min=0.1, max=self.max_weight)

        squared_error = (pred.flatten() - target.flatten()) ** 2
        return torch.mean(batch_weights * squared_error)
