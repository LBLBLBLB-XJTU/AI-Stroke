import torch
import torch.nn as nn

class CrossModalTokenFusion(nn.Module):
    """
    Cross-modal token fusion at each time step.
    No temporal mixing here.

    Input:
        skel_tokens:  (B, T, V_s, E)
        other_tokens: (B, T, V_o=1, E)

    Output:
        fused_tokens: (B, T, V_s + V_o, E)
    """
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        drop=0.2,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop,
        )
        self.attn_ln = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(drop),
        )
        self.ffn_ln = nn.LayerNorm(embed_dim)

    def forward(self, skel_tokens, other_tokens):
        """
        skel_tokens:  (B, T, V_s, E)
        other_tokens: (B, T, V_o, E)
        """
        B, T, V_s, E = skel_tokens.shape
        _, _, V_o, _ = other_tokens.shape

        # ------------------------------------------------
        # concat tokens at each time step
        # ------------------------------------------------
        tokens = torch.cat([skel_tokens, other_tokens], dim=2)
        # (B, T, V_s + V_o, E)

        V_total = V_s + V_o

        # ------------------------------------------------
        # per-time token attention (no temporal mixing)
        # ------------------------------------------------
        tokens_flat = tokens.view(B * T, V_total, E)

        attn_out, _ = self.attn(
            tokens_flat, tokens_flat, tokens_flat, need_weights=False
        )
        tokens_flat = self.attn_ln(tokens_flat + attn_out)

        # ------------------------------------------------
        # token-wise FFN
        # ------------------------------------------------
        ffn_out = self.ffn(tokens_flat)
        tokens_flat = self.ffn_ln(tokens_flat + ffn_out)

        fused_tokens = tokens_flat.view(B, T, V_total, E)

        return fused_tokens