"""
Loss functions for long-tailed / imbalanced classification.

Primary: BalancedSoftmaxLoss — adjusts class logits by log(class_frequency)
         so the softmax denominator is balanced across class sizes.
         Reference: "Balanced Meta-Softmax for Long-Tailed Visual Recognition" (NeurIPS 2020)

Secondary: LDAMLoss — label-distribution-aware margin pushes a larger decision
           margin for minority classes proportional to 1/sqrt(sqrt(N_j)).
           Reference: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss.

    Modifies the standard softmax by incorporating class frequency into the
    denominator so that more-frequent classes are penalised proportionally:

        p(y=c | x) = n_c * exp(f_c) / Σ_j n_j * exp(f_j)

    Equivalent to adding log(n_c) to each class logit before cross-entropy,
    which shifts decision boundaries in favour of minority classes.

    Args:
        cls_num_list: list/array of per-class sample counts (length = num_classes)
    """
    def __init__(self, cls_num_list):
        super().__init__()
        cls_num = torch.FloatTensor(cls_num_list)
        # log-frequency bias term added to logits
        self.register_buffer("log_freq", torch.log(cls_num / cls_num.sum()))

    def forward(self, logits, targets):
        # Shift logits by log-frequency to balance the softmax denominator
        adjusted = logits + self.log_freq.to(logits.device)
        return F.cross_entropy(adjusted, targets)


class LDAMLoss(nn.Module):
    """
    LDAM (Label-Distribution-Aware Margin) Loss.

    Imposes a class-specific margin m_j = C / sqrt(sqrt(N_j)) that is larger
    for minority classes, scaled by temperature s before the softmax.

    Args:
        cls_num_list: list/array of per-class sample counts
        max_m: maximum margin value (default 0.5)
        s: logit temperature scaling (default 30)
        weight: optional per-class weight tensor for cross-entropy
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.register_buffer("m_list", torch.FloatTensor(m_list))
        self.s = s
        self.weight = weight

    def forward(self, logits, targets):
        # Subtract margin from correct-class logit only
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        m = self.m_list.to(logits.device)
        batch_m = m[targets].unsqueeze(1)           # (B, 1)
        x_m = logits.clone()
        x_m[index] -= batch_m.squeeze(1)
        return F.cross_entropy(self.s * x_m, targets, weight=self.weight)
