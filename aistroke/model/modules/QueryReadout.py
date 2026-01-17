import torch
import torch.nn as nn

class QueryReadout(nn.Module):
    """
    Query-based readout over spatio-temporal tokens.

    Input:  (B, N, E)   # N = K * V
    Output: (B, Q, E)
    """
    def __init__(
        self,
        embed_dim,
        num_queries=2,
        num_heads=4,
        drop=0.1,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # learnable queries
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, embed_dim) * 0.02
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=drop,
        )

        self.ln1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(drop),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, node_tokens):
        """
        node_tokens: (B, N, E)
        return: (B, Q, E)
        """
        B, N, E = node_tokens.shape

        # expand queries for batch
        q = self.queries.expand(B, -1, -1)   # (B, Q, E)

        # cross attention: query <- nodes
        q_attn, _ = self.cross_attn(
            q, node_tokens, node_tokens, need_weights=False
        )
        q = self.ln1(q + q_attn)

        # FFN
        q_ffn = self.ffn(q)
        q = self.ln2(q + q_ffn)

        return q
