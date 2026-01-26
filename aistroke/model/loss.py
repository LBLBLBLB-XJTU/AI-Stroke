import torch
import torch.nn as nn
import torch.nn.functional as F

class Losses(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ce_lambda = cfg.LOSS.CE_LAMBDA 
        self.l2_lambda = cfg.LOSS.L2_LAMBDA
        self.center_lambda = cfg.LOSS.CENTER_LAMBDA

        self.ce = nn.CrossEntropyLoss(label_smoothing=cfg.LOSS.LABEL_SMOOTHING)

        # Center Loss parameters
        self.feat_dim = cfg.MODEL.EMBED_DIM
        self.centers_for_lr = nn.Parameter(torch.randn(2, self.feat_dim))
        self.centers_for_global = nn.Parameter(torch.randn(2, self.feat_dim * 2))

    def forward(self, feats, logits, left_labels, right_labels, total_labels, has_lr, model=None):
        feat_left, feat_right, feat_global = feats
        logit_left, logit_right, logits_global = logits
        device = logits_global.device

        # =========================================================
        # 1. CE Loss
        # =========================================================
        if has_lr:
            loss_left = self.ce(logit_left, left_labels)
            loss_right = self.ce(logit_right, right_labels)

            left_prob = torch.softmax(logit_left, dim=1)[:, 1]
            right_prob = torch.softmax(logit_right, dim=1)[:, 1]
            final_prob = torch.clamp(left_prob * right_prob, 1e-6, 1 - 1e-6)
            final_prob_tensor = torch.stack([1 - final_prob, final_prob], dim=1)
            loss_global = F.nll_loss(torch.log(final_prob_tensor), total_labels)

            ce_loss = loss_left + loss_right + loss_global
        else:
            ce_loss = self.ce(logits_global, total_labels)
        total_loss = self.ce_lambda * ce_loss

        # =========================================================
        # 2. L2 regularization
        # =========================================================
        l2_reg_loss = torch.tensor(0., device=device)
        if model is not None and self.l2_lambda > 0:
            for name, param in model.named_parameters():
                if param.requires_grad and "weight" in name:
                    l2_reg_loss += torch.sum(param ** 2)
            total_loss += self.l2_lambda * l2_reg_loss

        # =========================================================
        # 3. Center Loss
        # =========================================================
        center_loss = torch.tensor(0., device=device)
        if self.center_lambda > 0:
            if has_lr:
                centers = self.centers_for_lr.to(device)
                all_feats = torch.cat([feat_left, feat_right], dim=0)
                all_labels = torch.cat([left_labels, right_labels], dim=0)

                mask = all_labels != -1
                if mask.any():
                    feats_valid = all_feats[mask]
                    labels_valid = all_labels[mask]
                    centers_batch = centers[labels_valid]

                    center_loss = 0.5 * torch.mean(torch.sum((feats_valid - centers_batch) ** 2, dim=1))
            else:
                centers = self.centers_for_global.to(device)
                centers_batch = centers[total_labels]
                center_loss = 0.5 * torch.mean(torch.sum((feat_global - centers_batch) ** 2, dim=1))
            total_loss += self.center_lambda * center_loss

        return {
            "total_loss": total_loss,
            
            "ce_loss": ce_loss,
            "l2_reg_loss": l2_reg_loss,
            "center_loss": center_loss,
        }