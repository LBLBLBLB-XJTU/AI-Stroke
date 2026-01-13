import torch
import os
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from .visualisation_and_record import plot_confusion_matrix, plot_roc_curve, plot_pr_curve, save_error_records, save_upper_body_skeleton_3d_cam
from .gradcam_skeleton import UpperBodySkeletonGradCAM

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
    CosFace 训练 / 测试循环（支持 Skeleton Grad-CAM）
    """
    is_train = (mode == "train" and optimizer is not None)
    collect_metrics = (mode == "test")

    net.train() if is_train else net.eval()

    # --- 统计指标 ---
    correct_final = correct_left = correct_right = 0
    total_samples = 0

    total_loss = total_loss_ce = total_loss_l2 = 0.0
    total_loss_feat = total_loss_center = total_loss_triplet = 0.0
    batch_count = 0

    all_labels, all_preds, all_probs = [], [], []
    FP_records, FN_records = [], []

    # Grad-CAM 只在 eval / test 用
    # cam_extractor = UpperBodySkeletonGradCAM(net) if not is_train else None

    for batch_idx, batch in enumerate(data_loader):
        huanz_ids = batch[0]
        left_labels = batch[1].to(device)
        right_labels = batch[2].to(device)
        total_labels = batch[3].to(device)
        inputs = {k: v.to(device) for k, v in batch[4].items()}

        # ======================================================
        # 1️⃣ Forward（train: 有梯度；eval: 无梯度）
        # ======================================================
        with torch.enable_grad() if is_train else torch.no_grad():
            feats, _ = net(inputs)
            logit_left = net.left_head(
                feats[0], left_labels if is_train else None
            )
            logit_right = net.right_head(
                feats[1], right_labels if is_train else None
            )

            loss_dict = criterion(
                feats,
                (logit_left, logit_right),
                left_labels,
                right_labels,
                total_labels,
                model=net
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
        # 3️⃣ Grad-CAM（eval / test，单独 enable_grad）
        # ======================================================
        # if not is_train and save_dir is not None:
        #     with torch.enable_grad():
        #         cam_left = cam_extractor(inputs, side="left")
        #         cam_right = cam_extractor(inputs, side="right")

        #     # 只可视化 batch 中第一个样本
        #     t_vis = inputs["joints"].shape[1] // 2
        #     joints = inputs["joints"][0, t_vis].cpu().numpy()

        #     vis_dir = os.path.join(save_dir, "cam_vis", str(huanz_ids[0]))
        #     os.makedirs(vis_dir, exist_ok=True)

        #     save_upper_body_skeleton_3d_cam(
        #         joints_3d=joints,
        #         cam=cam_left[0].cpu().numpy(),
        #         save_path=os.path.join(vis_dir, "left_cam.png"),
        #         title="Left branch Grad-CAM"
        #     )
        #     save_upper_body_skeleton_3d_cam(
        #         joints_3d=joints,
        #         cam=cam_right[0].cpu().numpy(),
        #         save_path=os.path.join(vis_dir, "right_cam.png"),
        #         title="Right branch Grad-CAM"
        #     )

        # ======================================================
        # 4️⃣ Prediction & metrics
        # ======================================================
        left_pred = torch.argmax(logit_left, 1)
        right_pred = torch.argmax(logit_right, 1)
        final_pred = (left_pred & right_pred)

        bs = total_labels.numel()
        correct_final += (final_pred == total_labels).sum().item()
        correct_left += (left_pred == left_labels).sum().item()
        correct_right += (right_pred == right_labels).sum().item()
        total_samples += bs

        total_loss += loss_dict["total_loss"].item()
        total_loss_ce += loss_dict["ce_loss"].item()
        total_loss_l2 += loss_dict["l2_reg_loss"].item()
        total_loss_feat += loss_dict["feat_reg_loss"].item()
        total_loss_center += loss_dict["center_loss"].item()
        batch_count += 1

        # ======================================================
        # 5️⃣ 测试指标 & FP/FN
        # ======================================================
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

    # ======================================================
    # 6️⃣ 汇总
    # ======================================================
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

        "stage2_correct": correct_final
    }

    if collect_metrics:
        all_labels = torch.cat(all_labels).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_probs = np.concatenate(all_probs)

        results.update({
            "precision": precision_score(all_labels, all_preds),
            "recall": recall_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds),
            "auc": roc_auc_score(all_labels, all_probs)
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