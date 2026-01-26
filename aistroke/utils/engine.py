import torch
import os
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from .visualisation_and_record import plot_confusion_matrix, plot_roc_curve, plot_pr_curve, save_error_records

def safe_val(x, idx):
    return None if x is None else int(x[idx].item())

def safe_prob(x, idx):
    return None if x is None else float(x[idx].item())

def run_one_epoch(
    net,
    data_loader,
    device,
    criterion,
    optimizer=None,
    mode="eval",
    save_dir=None
):
    is_train = (mode == "train" and optimizer is not None)
    collect_metrics = (mode == "test")

    net.train() if is_train else net.eval()

    # --- 统计指标 ---
    correct_final = correct_left = correct_right = 0
    total_samples = total_left_samples = total_right_samples = 0

    total_loss = total_loss_ce = total_loss_l2 = total_loss_center = 0.0
    batch_count = 0

    all_labels, all_preds, all_probs = [], [], []
    FP_records, FN_records = [], []

    for batch_idx, batch in enumerate(data_loader):
        huanz_ids = batch["huanz_ids"]
        left_labels = batch["left_labels"].to(device)
        right_labels = batch["right_labels"].to(device)
        total_labels = batch["total_labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        has_lr = (left_labels != -1).all() and (right_labels != -1).all()

        # ======================================================
        # 1️⃣ Forward（train: 有梯度；eval: 无梯度）
        # ======================================================
        with torch.enable_grad() if is_train else torch.no_grad():
            feats, (logit_left, logit_right, logit_total) = net(inputs)

            loss_dict = criterion(
                feats,
                (logit_left, logit_right, logit_total),
                left_labels,
                right_labels,
                total_labels,
                has_lr,
                model=net,
            )

        # ======================================================
        # 2️⃣ Backward（仅训练）
        # ======================================================
        if is_train:
            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

        # ======================================================
        # 4️⃣ Prediction & metrics
        # ======================================================
        if has_lr:
            left_pred = torch.argmax(logit_left, 1)
            right_pred = torch.argmax(logit_right, 1)
            final_pred = (left_pred & right_pred)
        else:
            left_pred = right_pred = None
            final_pred = torch.argmax(logit_total, 1)

        bs = total_labels.numel()
        if has_lr:
            correct_left += (left_pred == left_labels).sum().item()
            correct_right += (right_pred == right_labels).sum().item()
            total_left_samples += bs
            total_right_samples += bs
        correct_final += (final_pred == total_labels).sum().item()
        total_samples += bs

        total_loss += loss_dict["total_loss"].item()
        total_loss_ce += loss_dict["ce_loss"].item()
        total_loss_l2 += loss_dict["l2_reg_loss"].item()
        total_loss_center += loss_dict["center_loss"].item()
        batch_count += 1

        # ======================================================
        # 5️⃣ 测试指标 & FP/FN
        # ======================================================
        if collect_metrics:
            all_labels.append(total_labels.cpu())
            all_preds.append(final_pred.cpu())

            if has_lr:
                left_prob = torch.softmax(logit_left, 1)[:, 1]
                right_prob = torch.softmax(logit_right, 1)[:, 1]
                final_prob = left_prob * right_prob
            else:
                left_prob = right_prob = None
                final_prob = torch.softmax(logit_total, 1)[:, 1]
            all_probs.append(final_prob.cpu())

            # ---- 记录 FP / FN 详细信息 ----
            FP_mask = (final_pred == 1) & (total_labels == 0)
            FN_mask = (final_pred == 0) & (total_labels == 1)

            for idx in torch.where(FP_mask)[0]:
                FP_records.append({
                    "huanz_id": huanz_ids[idx],
                    "final_pred": int(final_pred[idx].item()),
                    "final_prob": float(final_prob[idx].item()),
                    "total_label": int(total_labels[idx].item()),

                    "left_pred": safe_val(left_pred, idx),
                    "right_pred": safe_val(right_pred, idx),
                    "left_prob": safe_prob(left_prob, idx),
                    "right_prob": safe_prob(right_prob, idx),
                    "left_label": safe_val(left_labels if has_lr else None, idx),
                    "right_label": safe_val(right_labels if has_lr else None, idx),
                })
            for idx in torch.where(FN_mask)[0]:
                FN_records.append({
                    "huanz_id": huanz_ids[idx],
                    "final_pred": int(final_pred[idx].item()),
                    "final_prob": float(final_prob[idx].item()),
                    "total_label": int(total_labels[idx].item()),

                    "left_pred": safe_val(left_pred, idx),
                    "right_pred": safe_val(right_pred, idx),
                    "left_prob": safe_prob(left_prob, idx),
                    "right_prob": safe_prob(right_prob, idx),
                    "left_label": safe_val(left_labels if has_lr else None, idx),
                    "right_label": safe_val(right_labels if has_lr else None, idx),
                })

    # ======================================================
    # 6️⃣ 汇总
    # ======================================================
    batch_count = max(batch_count, 1)
    total_samples = max(total_samples, 1)

    results = {
        "final_acc": correct_final / total_samples,
        "left_acc": (correct_left / total_left_samples if total_left_samples > 0 else None),
        "right_acc": (correct_right / total_right_samples if total_right_samples > 0 else None),

        "total_loss": total_loss / batch_count,
        "ce_loss": total_loss_ce / batch_count,
        "l2_reg_loss": total_loss_l2 / batch_count,
        "center_loss": total_loss_center / batch_count,

        "net_correct": correct_final
    }

    if collect_metrics:
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = np.concatenate(all_probs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            results.update({
                "precision": precision_score(all_labels, all_preds, zero_division=0),
                "recall": recall_score(all_labels, all_preds, zero_division=0),
                "f1": f1_score(all_labels, all_preds, zero_division=0),
                "auc": (
                    roc_auc_score(all_labels, all_probs)
                    if len(np.unique(all_labels)) > 1 else None
                )
            })

        # 绘图
        os.makedirs(save_dir, exist_ok=True)
        plot_confusion_matrix(
            confusion_matrix(all_labels, all_preds),
            os.path.join(save_dir, "confusion_matrix.png")
        )
        plot_roc_curve(all_labels, all_probs, os.path.join(save_dir, "roc_curve.png"))
        plot_pr_curve(all_labels, all_probs, os.path.join(save_dir, "pr_curve.png"))
        save_error_records(
            FP_records, FN_records,
            os.path.join(save_dir, "error_records.json")
        )

    return results