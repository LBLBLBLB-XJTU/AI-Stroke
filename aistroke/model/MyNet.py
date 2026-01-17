import torch
import torch.nn as nn

from .modules.OtherModalTokenizer import OtherModalTokenizer
from .modules.SkeletonTokenizer import SkeletonTokenizer
from .modules.CrossModalTokenFusion import CrossModalTokenFusion
from .modules.TemporalModalAndPool import TemporalModelAndPool
from .modules.QueryReadout import QueryReadout
from .modules.AsymmetricDiffInteraction import AsymmetricDiffInteraction
from .modules.LinearHead import LinearHead

class MyNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # genreal config
        self.used_modalities = [cfg.MODEL.MODALITIES_NAMES[i] for i in cfg.MODEL.MODALITIES_USED_IDX]
        self.input_dims = [cfg.MODEL.MODALITIES_DIMS[i] for i in cfg.MODEL.MODALITIES_USED_IDX]
        self.embed_dim = cfg.MODEL.EMBED_DIM
        self.drop = cfg.MODEL.DROP
        self.conv_channels = cfg.MODEL.TOKEN_CONV_CHANNELS
        self.temporal_stride = cfg.MODEL.TEMPORAL_STRIDE
        self.num_heads = cfg.MODEL.NUM_HEADS
        self.max_T = cfg.MODEL.MAX_T
        # skeleton config
        self.skeleton_name = cfg.MODEL.SKELETON_NAME
        self.skel_in_channels = cfg.MODEL.SKELETON_CHANNELS
        self.skel_num_joints = cfg.MODEL.SKELETON_NUM_JOINTS
        # temporal config
        self.num_time_heads = cfg.MODEL.NUM_TIME_HEADS

        # ---- skeleton-specific tokenizer ----
        if self.skeleton_name in self.used_modalities:
            self.skeleton_tokenizer = SkeletonTokenizer(
                in_channels=self.skel_in_channels,
                embed_dim=self.embed_dim,
                num_joints=self.skel_num_joints,
                conv_channels=self.conv_channels,
                drop=self.drop,
                temporal_stride=self.temporal_stride,
                num_heads=self.num_heads,
                max_T=self.max_T
            )
            self.skeleton_gate = nn.Parameter(torch.tensor(1.0))

        # ---- tokenizers for non-skeleton modalities ----
        self.non_skel_tokenizers = nn.ModuleDict()
        for name, dim in zip(self.used_modalities, self.input_dims):
            if name == self.skeleton_name:
                continue
            self.non_skel_tokenizers[name] = OtherModalTokenizer(
                in_channels=dim, 
                embed_dim=self.embed_dim, 
                conv_channels=self.conv_channels, 
                drop=self.drop, 
                temporal_stride=self.temporal_stride,
                num_heads=self.num_heads,
                max_T=self.max_T)
        self.non_skel_names = [n for n in self.used_modalities if n != self.skeleton_name]
        self.modality_gate = nn.Parameter(torch.ones(len(self.non_skel_names)) * 1.0)  # length = num non-skel modalities

        self.cross_modal_fusion = CrossModalTokenFusion(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            drop=self.drop,
        )

        self.temoral_block = TemporalModelAndPool(
            embed_dim=self.embed_dim,  
            num_heads=self.num_heads,
            num_time_heads=self.num_time_heads,
            drop=self.drop,
        )

        self.query_block = QueryReadout(
            embed_dim=self.embed_dim,
            num_queries=2,
            num_heads=self.num_heads,
            drop=self.drop,
        )

        self.asmmy_diff_interaction = AsymmetricDiffInteraction(
            embed_dim=self.embed_dim,
            hidden_dim=self.embed_dim, 
            drop=self.drop,
        )
        
        self.feat_norm = nn.LayerNorm(self.embed_dim)
        self.head_left = LinearHead(
            embed_dim=self.embed_dim,
            num_classes=2,
        )
        self.head_right = LinearHead(
            embed_dim=self.embed_dim,
            num_classes=2, 
        )

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
        token_parts = []
        B = list(inputs.values())[0].shape[0]

        # ---- skeleton tokens ----
        if self.skeleton_name in self.used_modalities:
            sk = inputs[self.skeleton_name]
            node_tokens = self.skeleton_tokenizer(sk)
            node_tokens = node_tokens * (torch.sigmoid(self.skeleton_gate) * 1.2)

        # ---- other modalities ----
        if len(self.non_skel_names) > 0:
            gates = torch.sigmoid(self.modality_gate) * 1.2
            for i, name in enumerate(self.non_skel_names):
                x = inputs[name]
                if x.dim() > 3:
                    Bx, T, *rest = x.shape
                    x = x.view(Bx, T, -1)
                if x.dim() == 2:
                    x = x.unsqueeze(-1)
                token = self.non_skel_tokenizers[name](x)
                token = token * gates[i]
                token_parts.append(token)
            others_tokens = torch.cat(token_parts, dim=2)
        
        if node_tokens is not None and len(self.non_skel_names) == 0:
            dummy_other_tokens = torch.zeros(B, node_tokens.shape[1], 1, self.embed_dim, device=node_tokens.device)
            fused_tokens = self.cross_modal_fusion(skel_tokens=node_tokens, other_tokens=dummy_other_tokens)
        elif node_tokens is None and len(self.non_skel_names) > 0:
            dummy_skel_tokens = torch.zeros(B, others_tokens.shape[1], 1, self.embed_dim, device=others_tokens.device)
            fused_tokens = self.cross_modal_fusion(skel_tokens=dummy_skel_tokens, other_tokens=others_tokens)
        else:
            fused_tokens = self.cross_modal_fusion(skel_tokens=node_tokens, other_tokens=others_tokens)
        
        fused_tokens = self.temoral_block(fused_tokens)

        q_feats = self.query_block(fused_tokens)

        feat_left = q_feats[:, 0]
        feat_right = q_feats[:, 1]

        feat_left, feat_right = self.asmmy_diff_interaction(feat_left, feat_right)

        feat_left = self.feat_norm(feat_left)
        feat_right = self.feat_norm(feat_right)

        logits_left = self.head_left(feat_left)
        logits_right = self.head_right(feat_right)

        return (feat_left, feat_right), (logits_left, logits_right)