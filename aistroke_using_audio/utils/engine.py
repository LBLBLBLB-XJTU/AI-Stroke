import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import os
import numpy as np

# ======================
# 绘图函数
# ======================
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(labels, probs, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.savefig(save_path)
    plt.close()

def save_error_records(FP_records, FN_records, save_path):
    """
    将 FP/FN 详细记录保存为 JSON 文件
    """
    import json
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "FP": FP_records,
        "FN": FN_records
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run_one_epoch_cosface(
    net,
    data_loader,
    device,
    criterion,
    optimizer=None,
    mode="eval",
    save_dir=None
):
    """
    CosFace 训练/测试循环
    """
    is_train = (mode == "train" and optimizer is not None)
    collect_metrics = (mode == "test")

    net.train() if is_train else net.eval()

    # --- 统计指标 ---
    correct_final = 0
    correct_left = 0
    correct_right = 0
    total_samples = 0

    total_loss = total_loss_ce = total_loss_l2 = total_loss_feat = 0.0
    total_loss_center = total_loss_triplet = 0.0
    batch_count = 0

    all_labels, all_preds, all_probs = [], [], []

    # 新增：记录 FP / FN 的详细信息
    FP_records = []
    FN_records = []

    grad = torch.enable_grad() if is_train else torch.no_grad()

    with grad:
        for batch in data_loader:
            huanz_ids = batch[0]
            left_labels = batch[1].to(device)
            right_labels = batch[2].to(device)
            total_labels = batch[3].to(device)
            inputs = {k: v.to(device) for k, v in batch[4].items()}

            # ---- forward ----
            feats, _ = net(inputs)
            logit_left = net.left_head(feats[0], left_labels if is_train else None)
            logit_right = net.right_head(feats[1], right_labels if is_train else None)

            # ---- loss ----
            loss_dict = criterion(
                feats,
                (logit_left, logit_right),
                left_labels,
                right_labels,
                total_labels,
                model=net
            )

            if is_train:
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            # ---- prediction ----
            left_pred = torch.argmax(logit_left, 1)
            right_pred = torch.argmax(logit_right, 1)
            final_pred = (left_pred & right_pred)

            bs = total_labels.numel()
            correct_final += (final_pred == total_labels).sum().item()
            correct_left += (left_pred == left_labels).sum().item()
            correct_right += (right_pred == right_labels).sum().item()
            total_samples += bs

            # accumulate losses
            total_loss += loss_dict["total_loss"].item()
            total_loss_ce += loss_dict["ce_loss"].item()
            total_loss_l2 += loss_dict["l2_reg_loss"].item()
            total_loss_feat += loss_dict["feat_reg_loss"].item()
            total_loss_center += loss_dict["center_loss"].item()
            total_loss_triplet += loss_dict["triplet_loss"].item()
            batch_count += 1

            # ---- 测试指标收集 ----
            if collect_metrics:
                all_labels.append(total_labels.cpu())
                all_preds.append(final_pred.cpu())

                left_prob = torch.softmax(logit_left, 1)[:, 1]
                right_prob = torch.softmax(logit_right, 1)[:, 1]
                final_prob = left_prob * right_prob
                all_probs.append(final_prob.cpu())

                # ---- 记录 FP / FN 详细信息 ----
                FP_mask = (final_pred == 1) & (total_labels == 0)
                FN_mask = (final_pred == 0) & (total_labels == 1)

                for idx in torch.where(FP_mask)[0]:
                    FP_records.append({
                        "huanz_id": huanz_ids[idx],
                        "left_pred": int(left_pred[idx].item()),
                        "right_pred": int(right_pred[idx].item()),
                        "final_pred": int(final_pred[idx].item()),
                        "left_prob": float(left_prob[idx].item()),
                        "right_prob": float(right_prob[idx].item()),
                        "final_prob": float(final_prob[idx].item()),
                        "left_label": int(left_labels[idx].item()),
                        "right_label": int(right_labels[idx].item()),
                        "total_label": int(total_labels[idx].item()),
                    })

                for idx in torch.where(FN_mask)[0]:
                    FN_records.append({
                        "huanz_id": huanz_ids[idx],
                        "left_pred": int(left_pred[idx].item()),
                        "right_pred": int(right_pred[idx].item()),
                        "final_pred": int(final_pred[idx].item()),
                        "left_prob": float(left_prob[idx].item()),
                        "right_prob": float(right_prob[idx].item()),
                        "final_prob": float(final_prob[idx].item()),
                        "left_label": int(left_labels[idx].item()),
                        "right_label": int(right_labels[idx].item()),
                        "total_label": int(total_labels[idx].item()),
                    })

    # --- 汇总 ---
    batch_count = max(batch_count, 1)
    total_samples = max(total_samples, 1)

    results = {
        "final_acc": correct_final / total_samples,
        "left_acc": correct_left / total_samples,
        "right_acc": correct_right / total_samples,

        "total_loss": total_loss / batch_count,
        "ce_loss": total_loss_ce / batch_count,
        "l2_reg_loss": total_loss_l2 / batch_count,
        "feat_reg_loss": total_loss_feat / batch_count,
        "center_loss": total_loss_center / batch_count,
        "triplet_loss": total_loss_triplet / batch_count,
    }

    # --- 测试阶段计算额外指标 ---
    if collect_metrics:
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = np.concatenate(all_probs)

        results["precision"] = precision_score(all_labels, all_preds)
        results["recall"] = recall_score(all_labels, all_preds)
        results["f1"] = f1_score(all_labels, all_preds)
        try:
            results["auc"] = roc_auc_score(all_labels, all_probs)
        except:
            results["auc"] = 0.0

        # 加入 FP/FN 详细记录
        results["FP_records"] = FP_records
        results["FN_records"] = FN_records

        # 绘图
        cm = confusion_matrix(all_labels, all_preds)
        os.makedirs(save_dir, exist_ok=True)
        plot_confusion_matrix(cm, os.path.join(save_dir, "confusion_matrix.png"))
        plot_roc_curve(all_labels, all_probs, os.path.join(save_dir, "roc_curve.png"))
        plot_pr_curve(all_labels, all_probs, os.path.join(save_dir, "pr_curve.png"))
        save_error_records(FP_records, FN_records, os.path.join(save_dir, "error_records.json"))

    return results