import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFaceHead(nn.Module):
    def __init__(self, embed_dim, s, m):
        super().__init__()
        self.embed_dim = embed_dim
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(2, self.embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, feat, label=None):
        # feat: (B, E)
        # label: (B,) int64 tensor
        feat_norm = F.normalize(feat, dim=1)      # (B, E)
        W_norm = F.normalize(self.W, dim=1)      # (num_classes, E)
        cos_theta = feat_norm @ W_norm.t()       # (B, num_classes)
        if label is not None:
            one_hot = F.one_hot(label, num_classes=W_norm.size(0)).float()
            cos_theta = cos_theta - one_hot * self.m
        logits = cos_theta * self.s
        return logits