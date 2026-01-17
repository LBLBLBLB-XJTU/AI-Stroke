import torch
import torch.nn as nn

class AsymmetricDiffInteraction(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, drop=0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        self.diff_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )

        self.ln_l = nn.LayerNorm(embed_dim)
        self.ln_r = nn.LayerNorm(embed_dim)

    def forward(self, feat_l, feat_r):
        """
        feat_l, feat_r: (B, E)
        """
        # 显式差异
        diff = feat_l - feat_r          # (B, E)

        diff_feat = self.diff_mlp(diff) # (B, E)
        gate = self.gate(diff)          # (B, E)

        # 反对称更新
        feat_l_out = feat_l + gate * diff_feat
        feat_r_out = feat_r - gate * diff_feat

        return self.ln_l(feat_l_out), self.ln_r(feat_r_out)
