import numpy as np
import os
import os.path as osp
import joblib

CLIPPED_DATA_PATH = osp.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data_generate", "raw_label_data_clipped.pkl")
CLASS1_PATH = osp.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data_generate", "sample_txts", "class_1.txt")

def exclude_huanz(txt_path):
	with open(txt_path, "r", encoding="utf-8") as f:
		lines = f.read().split()

	result = [line.strip() for line in lines if line.strip()]
	return result

def assess_sample(sample, min_angle_threshold, max_diff_threshold):
    """
    输入：
        left_angles, right_angles: list of float
        left_seg, right_seg: tuple (start, end) 或 None
        min_angle_threshold: 最低抬手角度阈值
        max_diff_threshold: 双臂角度最大差阈值
    返回：
        status: "healthy" 或 "patient"
    """
    seg_flag = sample['clip_range'][2]
    if seg_flag != 2:
        return ("patient", "seg!=2")
    
    left_angles = sample['left_arm_angles_clipped']
    right_angles = sample['right_arm_angles_clipped']
    left_plateau = np.array(left_angles)
    right_plateau = np.array(right_angles)

    # 1. 最低角度判断
    if np.min(left_plateau) < min_angle_threshold or np.min(right_plateau) < min_angle_threshold:
        return ("patient", "angle<threshold")
    
    # 2. 双臂角度差判断
    # 对齐长度取最短的那一段
    L = min(len(left_plateau), len(right_plateau))
    diff = np.abs(left_plateau[:L] - right_plateau[:L])
    
    if np.max(diff) > max_diff_threshold:
        return ("patient", "diff>threshold")
    
    return ("healthy",0)

def stage1_rule_check(min_angle_threshold, max_diff_threshold, verbose=False):
    class_1_all = exclude_huanz(CLASS1_PATH)
    class_1_now = []
    classfied2wrong = []

    wrong_acc = 0
    data = joblib.load(CLIPPED_DATA_PATH)

    for sample in data:
        if sample['id'] in class_1_all:
            class_1_now.append(sample['id'])

        status = assess_sample(
            sample,
            min_angle_threshold=min_angle_threshold,
            max_diff_threshold=max_diff_threshold
        )

        if status[0] == "patient" and sample["total_label"] == 0:
            classfied2wrong.append((sample["id"], status[1]))
            wrong_acc += 1
        elif status[0] == "patient" and sample['total_label'] == 1:
            classfied2wrong.append((sample["id"], status[1]))

    # ---------- 指标 ----------
    class1_detected_id = [
        sid for sid, _ in classfied2wrong if sid in class_1_now
    ]

    class1_recall = (
        len(class1_detected_id) / len(class_1_now)
        if len(class_1_now) > 0 else 0
    )

    precision = (
        wrong_acc / len(classfied2wrong)
        if len(classfied2wrong) > 0 else 0
    )

    if verbose:
        print(f"min_angle={min_angle_threshold}, max_diff={max_diff_threshold}")
        print(f"  class1检出率: {class1_recall:.4f}")
        print(f"  错误判断准确率: {precision:.4f}")

    return class1_recall, precision


if __name__ == "__main__":
    min_angle_list = range(5, 50, 1)    # 20,25,...,60
    max_diff_list  = range(0, 30, 1)     # 5,10,...,40

    # best_recall = -1
    # best_params_recall = None

    # for min_angle in min_angle_list:
    #     for max_diff in max_diff_list:
    #         recall, precision = stage1_rule_check(
    #             min_angle, max_diff, verbose=False
    #         )

    #         if recall > best_recall:
    #             best_recall = recall
    #             best_params_recall = (min_angle, max_diff)

    # print("====== Test A：class1 检出率最优 ======")
    # print(f"最佳 min_angle_threshold = {best_params_recall[0]}")
    # print(f"最佳 max_diff_threshold  = {best_params_recall[1]}")
    # print(f"class1 检出率 = {best_recall:.4f}")

    # best_precision = -1
    # best_params_precision = None

    # for min_angle in min_angle_list:
    #     for max_diff in max_diff_list:
    #         recall, precision = stage1_rule_check(
    #             min_angle, max_diff, verbose=False
    #         )

    #         if precision > best_precision:
    #             best_precision = precision
    #             best_params_precision = (min_angle, max_diff)

    # print("====== Test B：错误判断准确率最优 ======")
    # print(f"最佳 min_angle_threshold = {best_params_precision[0]}")
    # print(f"最佳 max_diff_threshold  = {best_params_precision[1]}")
    # print(f"错误判断准确率 = {best_precision:.4f}")
    recall_threshold = 0.90

    best_precision = -1
    best_params = None

    for min_angle in min_angle_list:
        for max_diff in max_diff_list:
            recall, precision = stage1_rule_check(
                min_angle, max_diff, verbose=False
            )

            if recall >= recall_threshold:
                if precision > best_precision:
                    best_precision = precision
                    best_params = (min_angle, max_diff, recall)

    print("====== 推荐阈值（约束优化） ======")
    print(f"min_angle_threshold = {best_params[0]}")
    print(f"max_diff_threshold  = {best_params[1]}")
    print(f"class1_recall       = {best_params[2]:.4f}")
    print(f"precision           = {best_precision:.4f}")
# recall_threshold = 0.95
# ====== 推荐阈值（约束优化） ======
# min_angle_threshold = 43
# max_diff_threshold  = 12
# class1_recall       = 0.9518
# precision           = 0.8547