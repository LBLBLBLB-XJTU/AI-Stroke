import numpy as np
import os
import os.path as osp

CLIPPED_DATA_PATH = osp.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data_generate", "raw_label_data_clipped_byaudio.pkl")

def get_info_from_txt(txt_path):
	with open(txt_path, "r", encoding="utf-8") as f:
		lines = f.read().split()

	result = [line.strip() for line in lines if line.strip()]
	return result

def assess_sample(sample, min_angle_threshold, max_diff_threshold):
    left_angles = np.array(sample['left_arm_angles'])
    right_angles = np.array(sample['right_arm_angles'])

    left_mean = np.mean(left_angles)
    right_mean = np.mean(right_angles)

    # 1. 角度判断
    if left_mean < min_angle_threshold or right_mean < min_angle_threshold:
        return ("patient", "mean_angle<threshold")
    
    # 2. diff判断
    # 对齐长度取最短的那一段
    L = min(len(left_angles), len(right_angles))
    diff = np.abs(left_angles[:L] - right_angles[:L])
    diff_mean = np.mean(diff)
    
    if diff_mean > max_diff_threshold:
        return ("patient", "mean_diff>threshold")
    
    return ("healthy",0)

def test_one_setting(data, class_1_all,
                     min_angle_threshold,
                     max_diff_threshold,
                     verbose=False):
    """
    Stage1 规则在给定阈值下的评估
    约定：
        Positive = 正常
        Negative = 异常

        tp = 正常 → 判正常
        fp = 异常 → 判正常
        fn = 正常 → 判异常（致命）
        tn = 异常 → 判异常
    """

    # --- 计数 ---
    tp = 0  # 正常 → 判正常
    fp = 0  # 异常 → 判正常
    fn = 0  # 正常 → 判异常（致命）
    tn = 0  # 异常 → 判异常

    class1_total = 0        # 明显异常总数
    class1_detected = 0     # 被 Stage1 抓到的明显异常数

    for sample in data:
        gt_is_normal = (sample["total_label"] == 1)
        gt_is_abnormal = not gt_is_normal

        is_class1 = sample["id"] in class_1_all
        if is_class1:
            class1_total += 1

        status, reason = assess_sample(
            sample,
            min_angle_threshold=min_angle_threshold,
            max_diff_threshold=max_diff_threshold
        )

        pred_is_abnormal = (status == "patient")
        pred_is_normal = not pred_is_abnormal

        # --- 混淆矩阵（严格按你的定义） ---
        if gt_is_normal and pred_is_normal:
            tp += 1
        elif gt_is_abnormal and pred_is_normal:
            fp += 1
        elif gt_is_normal and pred_is_abnormal:
            fn += 1
        elif gt_is_abnormal and pred_is_abnormal:
            tn += 1

        # --- 明显异常检出 ---
        if gt_is_abnormal and pred_is_abnormal and is_class1:
            class1_detected += 1

    # --- 指标 ---
    class1_recall = (
        class1_detected / class1_total
        if class1_total > 0 else 0.0
    )

    # 被判“异常”的集合里，真正异常的比例
    abnormal_purity = (
        tn / (tn + fn)
        if (tn + fn) > 0 else 0.0
    )

    # 正常被误杀的比例（系统红线）
    false_kill_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0 else 0.0
    )

    metrics = {
        "class1_recall": class1_recall,
        "abnormal_purity": abnormal_purity,
        "false_kill_rate": false_kill_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    if verbose:
        print(f"min_angle={min_angle_threshold}, max_diff={max_diff_threshold}")
        print(f"  class1_recall      : {class1_recall:.4f}")
        print(f"  abnormal_purity    : {abnormal_purity:.4f}")
        print(f"  false_kill_rate    : {false_kill_rate:.4f}")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    return metrics

def stage1_train(idx, raw_data, cfg):
    CLASS1_PATH = os.path.join(os.path.dirname(cfg.PROJECT_ROOT), cfg.PATH.CLASS1_PATH)
    class_1_all = get_info_from_txt(CLASS1_PATH)

    data = [raw_data[i] for i in idx]

    min_angle_list = range(5, 60, 1)
    max_diff_list  = range(0, 40, 1)

    # ===== 可调超参数（系统级）=====
    purity_min = 0.95   # 正常样本误杀容忍度
    # ==============================

    best_recall = -1
    best_candidates = []

    for min_angle in min_angle_list:
        for max_diff in max_diff_list:
            metrics = test_one_setting(
                data, class_1_all,
                min_angle, max_diff,
                verbose=False
            )

            purity = metrics["abnormal_purity"]
            recall = metrics["class1_recall"]

            # --- 核心筛选逻辑 ---
            if purity >= purity_min:
                if recall > best_recall:
                    best_recall = recall
                    best_candidates = [(min_angle, max_diff, metrics)]
                elif recall == best_recall:
                    best_candidates.append((min_angle, max_diff, metrics))

    if not best_candidates:
        raise RuntimeError(
            f"No valid Stage1 params found with abnormal_purity >= {purity_min}"
        )

    # 保守策略：min_angle 最大
    best_params = max(best_candidates, key=lambda x: x[0])
    min_angle, max_diff, metrics = best_params

    print("====== 推荐 Stage1 阈值（高置信异常检测） ======")
    print(f"min_angle_threshold = {min_angle}")
    print(f"max_diff_threshold  = {max_diff}")
    print(f"class1_recall       = {metrics['class1_recall']:.4f}")
    print(f"abnormal_purity     = {metrics['abnormal_purity']:.4f}")
    print(f"false_kill_rate     = {metrics['false_kill_rate']:.4f}")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")

    # plot_stage1_tradeoff_scatter(
    #     data,
    #     class_1_all,
    #     min_angle_list,
    #     max_diff_list,
    #     chosen_params=(min_angle, max_diff),
    #     purity_threshold=purity_min,
    #     save_path=os.path.join(cfg.PROJECT_ROOT, "stage1_tradeoff.png")
    # )

    return min_angle, max_diff, metrics

def stage1_test(idx, raw_data, best_params):
    min_angle = best_params[0]
    max_diff = best_params[1]

    stage1_correct = 0
    idx_needed_net = []
    for id in idx:
        sample = raw_data[id]
        status = assess_sample(sample, min_angle_threshold=min_angle, max_diff_threshold=max_diff)
        if status[0] == "patient" and sample["total_label"] == 0:
                stage1_correct += 1
        if status[0] == "healthy":
            idx_needed_net.append(id)

    return stage1_correct, idx_needed_net

import matplotlib.pyplot as plt
import os

def plot_stage1_tradeoff_scatter(
    data,
    class_1_all,
    min_angle_list,
    max_diff_list,
    chosen_params=None,
    purity_threshold=0.95,
    save_path="stage1_tradeoff.png"
):
    """
    可视化 Stage1 阈值选择的合理性：
    x: class1_recall
    y: abnormal_purity
    """

    recalls = []
    purities = []
    false_kills = []

    records = []

    for min_angle in min_angle_list:
        for max_diff in max_diff_list:
            metrics = test_one_setting(
                data,
                class_1_all,
                min_angle,
                max_diff,
                verbose=False
            )
            recalls.append(metrics["class1_recall"])
            purities.append(metrics["abnormal_purity"])
            false_kills.append(metrics["false_kill_rate"])
            records.append((min_angle, max_diff, metrics))

    plt.figure(figsize=(8, 6))

    # --- 所有候选点 ---
    sc = plt.scatter(
        recalls,
        purities,
        c=false_kills,
        cmap="viridis",
        alpha=0.6,
        s=25
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("False Kill Rate (Normal → Abnormal)")

    # --- purity 阈值线 ---
    plt.axhline(
        y=purity_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Purity ≥ {purity_threshold}"
    )

    # --- 高亮最终选择 ---
    if chosen_params is not None:
        min_angle, max_diff = chosen_params
        for ma, md, m in records:
            if ma == min_angle and md == max_diff:
                plt.scatter(
                    m["class1_recall"],
                    m["abnormal_purity"],
                    s=120,
                    c="red",
                    edgecolors="black",
                    marker="*",
                    label=f"Chosen ({ma}, {md})"
                )
                break

    plt.xlabel("Class1 Recall (Obvious Abnormal Detection)")
    plt.ylabel("Abnormal Purity (Precision of Stage1)")
    plt.title("Stage1 Threshold Trade-off Analysis")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Stage1 trade-off plot saved to {save_path}")