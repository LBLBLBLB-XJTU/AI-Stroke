import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalTokenizer(nn.Module):
    """
    A skeleton-style tokenizer for non-skeleton temporal signals.
    Input:  (B, T, C_in)
    Output: (B, token_len, embed_dim)
    Treat V = 1 as a single joint.
    """
    def __init__(self, input_dim, embed_dim,
                 conv_channels=[64,128], 
                 drop=0.2, T_pool=64):
        super().__init__()
        self.V = 1   # treat as skeleton with V=1
        self.embed_dim = embed_dim
        self.T_pool = T_pool
        
        # Step1: Temporal downsample (T -> T_pool)
        self.temporal_prepool = nn.AdaptiveAvgPool2d((self.T_pool, self.V))
        # Step2: Progressive Conv2d (C -> E)
        layers = []
        prev_ch = input_dim
        for ch in conv_channels:
            layers.append(nn.Conv2d(prev_ch, ch, kernel_size=(3,1), stride=(2,1), padding=(1,0)))
            layers.append(nn.GELU())
            layers.append(nn.GroupNorm(1, ch))
            layers.append(nn.Dropout(drop))
            prev_ch = ch
        self.temporal_conv = nn.Sequential(*layers)
        # Step3: Pool T -> token_len
        self.time_pool = nn.AdaptiveAvgPool2d((1, self.V))
        # Step4: final conv to embed_dim
        if prev_ch != embed_dim:
            self.final_conv = nn.Sequential(
                nn.Conv2d(prev_ch, embed_dim, kernel_size=(1,1)),
                nn.GELU()
            )
        else:
            self.final_conv = nn.Identity()
        
        # Positional encoding for V=1
        self.joint_pos = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Since V=1, spatial attention becomes identity
        self.norm = nn.LayerNorm(embed_dim)
        
        # Node FFN
        self.node_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim*2, embed_dim),
            nn.Dropout(drop)
        )
        self.node_ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, T, C_in)
        output: (B, token_len, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        B, T, C = x.shape
        # reshape to (B, C, T, V=1)
        h = x.permute(0,2,1).unsqueeze(-1).contiguous()

        # Step1: prepool for long sequences
        if T > self.T_pool:
            h = self.temporal_prepool(h)  # (B, C, T_pool, 1)

        # Step2: conv2d
        h = self.temporal_conv(h)        # (B, Ch, T', 1)

        # Step3: time pooling to token_len
        h = self.time_pool(h)            # (B, Ch, token_len, 1)
        # Step4: final conv to embed_dim
        h = self.final_conv(h)           # (B, E, token_len, 1)

        # squeeze and reshape: (B, token_len, E)
        h = h.squeeze(-1).permute(0,2,1).contiguous()

        # add pos encoding
        h = h + self.joint_pos           # (1,1,E)

        # no attention needed (V=1)
        h = self.norm(h)

        # FFN
        h_ffn = self.node_ffn(h)
        h = self.node_ln(h + h_ffn)
        return h