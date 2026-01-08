import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonTokenizer(nn.Module):
    """
    Skeleton tokenizer for large T (100~1000+):
    - Input: (B, T, V, C)
    - Output: (B, V, E)
    - Step1: AdaptiveAvgPool2d(T_pool) to reduce large T
    - Step2: Progressive Conv2d to raise channels C->E
    - Step3: Final AdaptiveAvgPool2d to collapse T -> 1
    """
    def __init__(self, in_channels=3, embed_dim=128, num_joints=11,
                 conv_channels=[64,128], drop=0.2, num_heads=4,
                 T_pool=32):
        super().__init__()
        self.V = num_joints
        self.embed_dim = embed_dim
        self.T_pool = T_pool
        
        # Step1: Adaptive temporal pooling for very large T
        self.temporal_prepool = nn.AdaptiveAvgPool2d((self.T_pool, self.V))
        # Step2: Progressive Conv2d (C -> E)
        layers = []
        prev_ch = in_channels
        for ch in conv_channels:
            layers.append(nn.Conv2d(prev_ch, ch, kernel_size=(3,1), stride=(2,1), padding=(1,0)))
            layers.append(nn.GELU())
            layers.append(nn.GroupNorm(1, ch)) # 或许需要改回Layernorm
            layers.append(nn.Dropout(drop))
            prev_ch = ch
        self.temporal_conv = nn.Sequential(*layers)
        # Step3: Final AdaptiveAvgPool2d to collapse T -> 1
        self.final_pool = nn.AdaptiveAvgPool2d((1, self.V))
        # Optional: final conv to embed_dim if last conv_channels[-1] != embed_dim
        if prev_ch != embed_dim:
            self.final_conv = nn.Sequential(
                nn.Conv2d(prev_ch, embed_dim, kernel_size=(1,1)),
                nn.GELU()
            )
        else:
            self.final_conv = nn.Identity()
        
        # joint positional encoding
        self.joint_pos = nn.Parameter(torch.randn(1, self.V, embed_dim) * 0.02)
        
        # spatial attention
        self.spatial_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                                  batch_first=True, dropout=drop)
        self.norm = nn.LayerNorm(embed_dim)
        
        # node-level FFN
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
        x: (B, T, V, C)
        return: (B, V, E)
        """
        B, T, V, C = x.shape
        assert V == self.V
        # permute to (B, C, T, V)
        h = x.permute(0, 3, 1, 2).contiguous()
        
        # Step1: temporal prepool for very large T
        if T > self.T_pool:
            h = self.temporal_prepool(h)
        
        # Step2: progressive conv
        h = self.temporal_conv(h)
        
        # Step3: final pooling to collapse T -> 1
        h = self.final_pool(h)
        h = self.final_conv(h)
        h = h.squeeze(2)  # (B, E, V)
        
        # transpose to (B, V, E)
        h = h.permute(0, 2, 1).contiguous()
        
        # add joint positional encoding
        h = h + self.joint_pos
        
        # spatial attention
        h_attn, _ = self.spatial_attn(h, h, h, need_weights=False)
        h = self.norm(h + h_attn)
        
        # node-level FFN
        h_ffn = self.node_ffn(h)
        h = self.node_ln(h + h_ffn)
        return h