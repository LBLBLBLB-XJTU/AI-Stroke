import torch
import torch.nn as nn

from .modules.TemporalTokenizer import TemporalTokenizer
from .modules.SkeletonTokenizer import SkeletonTokenizer
from .modules.CosFaceHead import CosFaceHead

class MyNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ---- config ----
        self.used_modalities = [cfg.MODEL.MODALITIES_NAMES[i] for i in cfg.MODEL.MODALITIES_USED_IDX]
        self.input_dims = [cfg.MODEL.MODALITIES_DIMS[i] for i in cfg.MODEL.MODALITIES_USED_IDX]
        self.embed_dim = int(cfg.MODEL.EMBED_DIM )
        self.fusion_layers = int(cfg.MODEL.TRANSFORMER_LAYERS)
        self.drop = float(cfg.MODEL.DROP)
        self.num_heads = int(cfg.MODEL.NUM_HEADS)

        # skeleton config
        self.skeleton_name = cfg.MODEL.SKELETON_NAME
        self.num_joints = int(cfg.MODEL.NUM_JOINTS)
        self.skel_in_channels = int(cfg.MODEL.SKELETON_CHANNELS)
        self.skel_num_heads = int(cfg.MODEL.SKELETON_NUM_HEADS)
        self.skel_conv_channels = cfg.MODEL.SKELETON_CONV_CHANNELS
        self.skel_time_pool_to = int(getattr(cfg.MODEL, "SKELETON_TIME_POOL_TO", 32))

        # other modal config
        self.other_conv_channels = cfg.MODEL.OTHER_CONV_CHANNELS
        self.other_time_pool_to = cfg.MODEL.OTHER_TIME_POOL_TO

        # classifer
        self.face_s = cfg.MODEL.FACE_S
        self.face_m = cfg.MODEL.FACE_M

        # ---- skeleton-specific tokenizer ----
        if self.skeleton_name in self.used_modalities:
            self.skeleton_tokenizer = SkeletonTokenizer(
                in_channels=self.skel_in_channels,
                embed_dim=self.embed_dim,
                num_joints=self.num_joints,
                conv_channels=self.skel_conv_channels,
                drop=self.drop,
                num_heads=self.skel_num_heads,
                T_pool=self.skel_time_pool_to
            )
            self.skeleton_gate = nn.Parameter(torch.tensor(1.0))

        # ---- tokenizers for non-skeleton modalities ----
        self.non_skel_tokenizers = nn.ModuleDict()
        for name, dim in zip(self.used_modalities, self.input_dims):
            if name == self.skeleton_name:
                continue
            self.non_skel_tokenizers[name] = TemporalTokenizer(
                input_dim=dim, 
                embed_dim=self.embed_dim, 
                conv_channels=self.other_conv_channels, 
                drop=self.drop, 
                T_pool=self.other_time_pool_to)
        self.non_skel_names = [n for n in self.used_modalities if n != self.skeleton_name]
        self.modality_gate = nn.Parameter(torch.ones(len(self.non_skel_names)) * 1.0)  # length = num non-skel modalities

        # ---- fusion transformer ----
        self.fusion_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True, dropout=self.drop)

        # ---- left/right query heads ----
        self.query_left = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.query_right = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        self.attn_qk = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True, dropout=self.drop)
        self.query_self_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=max(1, self.num_heads // 2), batch_first=True, dropout=self.drop)

        self.q_ln1 = nn.LayerNorm(self.embed_dim)
        self.q_ff = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(self.drop)
        )
        self.q_ln2 = nn.LayerNorm(self.embed_dim)
        self.feat_norm = nn.LayerNorm(self.embed_dim)

        self.left_head = CosFaceHead(self.embed_dim, self.face_s, self.face_m)
        self.right_head = CosFaceHead(self.embed_dim, self.face_s, self.face_m)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                # Conv weight: Kaiming
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        node_tokens = None
        token_parts = []  # list of (B, S_i, E)

        # ---- skeleton tokens ----
        if self.skeleton_name in self.used_modalities:
            sk = inputs[self.skeleton_name]
            B = sk.shape[0]
            node_tokens = self.skeleton_tokenizer(sk)  # (B, V, E)
            node_tokens = node_tokens * (torch.sigmoid(self.skeleton_gate) * 1.2)

        # ---- other modalities ----
        if len(self.non_skel_names) > 0:
            gates = torch.sigmoid(self.modality_gate) * 1.2
            for i, name in enumerate(self.non_skel_names):
                x = inputs[name]
                B = x.shape[0]
                if x.dim() > 3:
                    Bx, T, *rest = x.shape
                    x = x.view(Bx, T, -1)
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
                t = self.non_skel_tokenizers[name](x)  # (B, token_len, E)
                t = t * gates[i]
                token_parts.append(t)

        if node_tokens is not None and token_parts != []:
            tokens = torch.cat(token_parts, dim=1)
            fused, _ = self.fusion_attn(node_tokens, tokens, tokens)
            fused = node_tokens + fused
        elif node_tokens is not None and token_parts == []:
            fused, _ = self.fusion_attn(node_tokens, node_tokens, node_tokens)
            fused = node_tokens + fused
        elif node_tokens is None and token_parts != []:
            tokens = torch.cat(token_parts, dim=1)
            fused, _ = self.fusion_attn(tokens, tokens, tokens)
            fused = tokens + fused

        # queries attend to fused tokens
        q_left = self.query_left.expand(B, -1, -1)    # (B,1,E)
        q_right = self.query_right.expand(B, -1, -1)  # (B,1,E)
        queries = torch.cat([q_left, q_right], dim=1)  # (B,2,E)

        # cross-attn: queries <- fused tokens (Q, K, V)
        q_attn, _ = self.attn_qk(queries, fused, fused, need_weights=False)  # (B,2,E)

        # query self-attn (left <-> right interaction)
        q_res = q_attn
        q_sa_out, _ = self.query_self_attn(q_res, q_res, q_res, need_weights=False)  # (B,2,E)
        q_res = self.q_ln1(q_res + q_sa_out)
        q_ff_out = self.q_ff(q_res)
        q_res = self.q_ln2(q_res + q_ff_out)  # (B,2,E)

        feat_left = q_res[:, 0, :]   # (B,E)
        feat_right = q_res[:, 1, :]  # (B,E)

        feat_left = self.feat_norm(feat_left)
        feat_right = self.feat_norm(feat_right)

        logit_left = self.left_head(feat_left)
        logit_right = self.right_head(feat_right)

        return (feat_left, feat_right), (logit_left, logit_right)