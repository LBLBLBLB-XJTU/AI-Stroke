import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.MODEL.MAIN_NET.HEAD.ARCFACE.EMBED_DIM
        self.s = cfg.MODEL.MAIN_NET.HEAD.ARCFACE.S
        self.m = cfg.MODEL.MAIN_NET.HEAD.ARCFACE.M
        self.W = nn.Parameter(torch.randn(2, self.embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, feat, label=None):
        # feat: (B, E)
        # label: (B,) int64 tensor
        feat_norm = F.normalize(feat, dim=1)      # (B, E)
        W_norm = F.normalize(self.W, dim=1)      # (num_classes, E)
        cos_theta = feat_norm @ W_norm.t()        # (B, num_classes)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)  # avoid numerical errors
        if label is not None:
            theta = torch.acos(cos_theta)
            theta_m = theta + self.m
            cos_theta_m = torch.cos(theta_m)
            one_hot = F.one_hot(label, num_classes=W_norm.size(0)).float()
            cos_theta = cos_theta * (1 - one_hot) + cos_theta_m * one_hot
        logits = cos_theta * self.s
        return logits
