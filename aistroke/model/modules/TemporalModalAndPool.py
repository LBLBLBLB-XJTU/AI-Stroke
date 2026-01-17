import torch
import torch.nn as nn

class TemporalModelAndPool(nn.Module):
    """
    Temporal modeling on T dimension, then collapse T via multi-head attention pooling.

    Input:  (B, T, V, E)
    Output: (B, K*V, E)
    """

    def __init__(self, embed_dim, num_heads=4, num_time_heads=4, drop=0.1):
        super().__init__()

        self.num_time_heads = num_time_heads

        # -------------------------------------------------
        # 1. Temporal self-attention (per joint / modality)
        # -------------------------------------------------
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop,
        )
        self.temporal_ln = nn.LayerNorm(embed_dim)

        # -------------------------------------------------
        # 2. Multi-head temporal attention pooling
        # -------------------------------------------------
        self.time_score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_time_heads)  # K heads
        )

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        x: (B, T, V, E)
        return: (B, K*V, E)
        """
        B, T, V, E = x.shape
        K = self.num_time_heads

        # -------------------------------------------------
        # 1. Temporal self-attention (no joint mixing)
        # -------------------------------------------------
        # (B, V, T, E) → (B*V, T, E)
        h = x.permute(0, 2, 1, 3).contiguous()
        h = h.view(B * V, T, E)

        attn_out, _ = self.temporal_attn(h, h, h, need_weights=False)
        h = self.temporal_ln(h + self.drop(attn_out))

        # back to (B, T, V, E)
        h = h.view(B, V, T, E).permute(0, 2, 1, 3).contiguous()

        # -------------------------------------------------
        # 2. Multi-head temporal attention pooling
        # -------------------------------------------------
        # score: (B, T, V, K)
        score = self.time_score(h)

        # softmax over time dimension
        alpha = torch.softmax(score, dim=1)
        alpha = self.drop(alpha)

        # weighted sum over time
        # (B, K, V, E)
        pooled = torch.einsum(
            "btvk,btve->bkve", alpha, h
        )

        # -------------------------------------------------
        # 3. Merge (K, V) → token dimension
        # -------------------------------------------------
        # (B, K*V, E)
        pooled = pooled.reshape(B, K * V, E)

        return pooled
