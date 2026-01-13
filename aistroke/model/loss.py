import torch
import torch.nn as nn
import torch.nn.functional as F

class Losses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ce_lambda = cfg.LOSS.CE_LAMBDA 
        self.l2_lambda = cfg.LOSS.L2_LAMBDA
        self.feat_lambda = cfg.LOSS.FEAT_LAMBDA
        self.center_lambda = cfg.LOSS.CENTER_LAMBDA
        self.triplet_lambda = cfg.LOSS.TRIPLET_LAMBDA
        self.margin = cfg.LOSS.TRIPLET_MARGIN

        self.ce = nn.CrossEntropyLoss(label_smoothing=cfg.LOSS.LABEL_SMOOTHING)

        # Center Loss parameters
        self.feat_dim = cfg.MODEL.EMBED_DIM
        self.centers = nn.Parameter(torch.randn(2, self.feat_dim))

    def forward(self, feats, logits, left_labels, right_labels, total_labels, model=None):
        feat_left, feat_right = feats
        logit_left, logit_right = logits

        # =========================================================
        # 1. CE Loss (原逻辑保持)
        # =========================================================
        loss_left = self.ce(logit_left, left_labels)
        loss_right = self.ce(logit_right, right_labels)

        left_prob = torch.softmax(logit_left, dim=1)[:, 1]
        right_prob = torch.softmax(logit_right, dim=1)[:, 1]
        final_prob = torch.clamp(left_prob * right_prob, 1e-6, 1 - 1e-6)

        final_prob_tensor = torch.stack([1 - final_prob, final_prob], dim=1)
        loss_global = F.nll_loss(torch.log(final_prob_tensor), total_labels)

        ce_loss = loss_left + loss_right + loss_global
        total_loss = self.ce_lambda * ce_loss

        # =========================================================
        # 2. L2 regularization
        # =========================================================
        l2_reg_loss = torch.tensor(0., device=logit_left.device)
        if model is not None and self.l2_lambda > 0:
            for name, param in model.named_parameters():
                if param.requires_grad and "weight" in name:
                    l2_reg_loss += torch.sum(param ** 2)
            total_loss += self.l2_lambda * l2_reg_loss

        # =========================================================
        # 3. Center Loss
        # =========================================================
        center_loss = torch.tensor(0., device=logit_left.device)
        if self.center_lambda > 0:

            all_feats = torch.cat([feat_left, feat_right], dim=0)
            all_labels = torch.cat([left_labels, right_labels], dim=0)

            # 正确方式：使用一个设备一致的副本，而不修改 Parameter 本身
            centers = self.centers.to(all_feats.device)

            centers_batch = centers[all_labels]
            center_loss = 0.5 * torch.mean(torch.sum((all_feats - centers_batch) ** 2, dim=1))

            total_loss += self.center_lambda * center_loss

        return {
            "total_loss": total_loss,

            "ce_loss": ce_loss,
            "l2_reg_loss": l2_reg_loss,
            "center_loss": center_loss,
        }
