import torch
import torch.nn as nn

class OtherModalTokenizer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        embed_dim=128,
        conv_channels=(64, 128),
        drop=0.2,
        temporal_stride=(2, 2),
        num_heads=4,
        max_T=512,
    ):
        super().__init__()

        self.V = 1
        self.embed_dim = embed_dim

        layers = []
        prev_ch = in_channels
        for i, ch in enumerate(conv_channels):
            layers.append(
                nn.Conv2d(
                    prev_ch,
                    ch,
                    kernel_size=(3, 1),
                    stride=(temporal_stride[i], 1),
                    padding=(1, 0),
                )
            )
            layers.append(nn.GELU())
            layers.append(nn.GroupNorm(1, ch))
            layers.append(nn.Dropout(drop))
            prev_ch = ch
        self.temporal_conv = nn.Sequential(*layers)

        self.channel_proj = (
            nn.Conv2d(prev_ch, embed_dim, kernel_size=1)
            if prev_ch != embed_dim
            else nn.Identity()
        )

        # joint positional encoding (shared across time)
        self.feat_pos = nn.Parameter(
            torch.randn(1, 1, self.V, embed_dim) * 0.02
        )

        # time positional encoding (shared across joints)
        self.time_pos = nn.Parameter(
            torch.randn(1, max_T, 1, embed_dim) * 0.02
        )

        self.feat_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop,
        )
        self.feat_attn_ln = nn.LayerNorm(embed_dim)

        self.node_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(drop),
        )
        self.node_ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, C)
        return: (B, T', V, E)
        """
        h = x.unsqueeze(-1).contiguous()

        # (B, C, T, V)
        h = h.permute(0, 2, 1, 3).contiguous()

        # temporal conv (downsample T)
        h = self.temporal_conv(h)          # (B, Ch, T', V)
        h = self.channel_proj(h)           # (B, E, T', V)

        # (B, T', V, E)
        h = h.permute(0, 2, 3, 1).contiguous()
        B, Tt, V, E = h.shape

        # -----------------------------
        # add positional encodings
        # -----------------------------
        h = h + self.feat_pos
        h = h + self.time_pos[:, :Tt]

        # -----------------------------
        # feature attention (no time mixing)
        # -----------------------------
        h_flat = h.view(B * Tt, V, E)

        attn_out, _ = self.feat_attn(
            h_flat, h_flat, h_flat, need_weights=False
        )
        h_flat = self.feat_attn_ln(h_flat + attn_out)

        h = h_flat.view(B, Tt, V, E)

        # -----------------------------
        # node FFN
        # -----------------------------
        h = self.node_ln(h + self.node_ffn(h))
        return h