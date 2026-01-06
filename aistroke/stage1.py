import numpy as np
import os
import os.path as osp

CLIPPED_DATA_PATH = osp.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data_generate", "raw_label_data_clipped_byaudio.pkl")
CLASS1_PATH = osp.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data_generate", "sample_txts", "class_1.txt")

def get_info_from_txt(txt_path):
	with open(txt_path, "r", encoding="utf-8") as f:
		lines = f.read().split()

	result = [line.strip() for line in lines if line.strip()]
	return result

def assess_sample(sample, min_angle_threshold, max_diff_threshold):
    clip_start, clip_end = sample["clip_range"]
    left_angles_clipped = sample['left_arm_angles'][clip_start:clip_end+1]
    right_angles_clipped = sample['right_arm_angles'][clip_start:clip_end+1]
    left_angles = np.array(left_angles_clipped)
    right_angles = np.array(right_angles_clipped)

    # 1. 最低角度判断
    if np.min(left_angles) < min_angle_threshold or np.min(right_angles) < min_angle_threshold:
        return ("patient", "angle<threshold")
    
    # 2. 双臂角度差判断
    # 对齐长度取最短的那一段
    L = min(len(left_angles), len(right_angles))
    diff = np.abs(left_angles[:L] - right_angles[:L])
    
    if np.max(diff) > max_diff_threshold:
        return ("patient", "diff>threshold")
    
    return ("healthy",0)

def test_one_setting(data, class_1_all, min_angle_threshold, max_diff_threshold, verbose=False):
    class_1_now = []
    classfied2wrong = []
    wrong_acc = 0

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
    # 明显患病的检出率
    class1_detected_id = [
        sid for sid, _ in classfied2wrong if sid in class_1_now
    ]
    class1_recall = (
        len(class1_detected_id) / len(class_1_now)
        if len(class_1_now) > 0 else 0
    )
    
    # 确实患病的正确率
    precision = (
        wrong_acc / len(classfied2wrong)
        if len(classfied2wrong) > 0 else 0
    )

    if verbose:
        print(f"min_angle={min_angle_threshold}, max_diff={max_diff_threshold}")
        print(f"  class1检出率: {class1_recall:.4f}")
        print(f"  错误判断准确率: {precision:.4f}")

    return class1_recall, precision

def stage1_train(idx, raw_data):
    class_1_all = get_info_from_txt(CLASS1_PATH)
    indices = list(idx)
    data = [raw_data[i] for i in indices]

    min_angle_list = range(5, 50, 1)    # 20,25,...,60
    max_diff_list  = range(0, 30, 1)     # 5,10,...,40

    # plot_threshold_heatmap_save_with_recall(
    # raw_data, 
    # class_1_all, 
    # min_angle_list, 
    # max_diff_list, 
    # recall_threshold=0.9,
    # save_path="threshold_heatmap.png"
    # )

    recall_threshold = 0.90
    best_precision = -1
    best_params = None
    best_candidates = []

    for min_angle in min_angle_list:
        for max_diff in max_diff_list:
            recall, precision = test_one_setting(data, class_1_all, min_angle, max_diff, verbose=False)

            if recall >= recall_threshold:
                if precision > best_precision:
                    best_precision = precision
                    best_recall = recall
                    best_params = (min_angle, max_diff, best_precision, best_recall)
                    best_candidates = [(min_angle, max_diff, recall, precision)]
                elif precision == best_precision:
                    best_candidates.append((min_angle, max_diff, recall, precision))
                    
    
    print("====== 推荐阈值（约束优化） ======")
    print(f"min_angle_threshold = {best_params[0]}")
    print(f"max_diff_threshold  = {best_params[1]}")
    print(f"class1_recall       = {best_recall:.4f}")
    print(f"precision           = {best_precision:.4f}")
    # best_params = max(best_candidates, key=lambda x: x[0])
    return best_params

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
import numpy as np
import os

def plot_threshold_heatmap_save_with_recall(data, class_1_all, min_angle_range, max_diff_range, recall_threshold=0.9, save_path="threshold_heatmap.png"):
    """
    画二维阈值–准确率热图，颜色表示 precision，叠加 recall >= recall_threshold 等值线
    并标明 recall >= threshold 区域
    """
    precision_matrix = np.zeros((len(max_diff_range), len(min_angle_range)))
    recall_matrix = np.zeros((len(max_diff_range), len(min_angle_range)))

    # 遍历每个阈值组合
    for i, max_diff in enumerate(max_diff_range):
        for j, min_angle in enumerate(min_angle_range):
            recall, precision = test_one_setting(data, class_1_all, min_angle, max_diff, verbose=False)
            precision_matrix[i, j] = precision
            recall_matrix[i, j] = recall

    min_angle_vals = list(min_angle_range)
    max_diff_vals = list(max_diff_range)

    plt.figure(figsize=(10, 6))
    
    # 热图：precision
    plt.imshow(
        precision_matrix, 
        origin='lower', 
        extent=[min_angle_vals[0], min_angle_vals[-1], max_diff_vals[0], max_diff_vals[-1]],
        aspect='auto',
        cmap='viridis'
    )
    plt.colorbar(label='Precision')
    plt.xlabel('min_angle_threshold')
    plt.ylabel('max_diff_threshold')
    plt.title('Precision Heatmap with Recall Constraint')

    # 填充 recall >= threshold 区域
    recall_mask = recall_matrix >= recall_threshold
    plt.contourf(
        min_angle_vals,
        max_diff_vals,
        recall_mask,
        levels=[0.5, 1],   # True=1, False=0
        colors=['none', 'red'],
        alpha=0.2           # 半透明
    )

    # 等值线
    CS = plt.contour(
        min_angle_vals,
        max_diff_vals,
        recall_matrix,
        levels=[recall_threshold],
        colors='red',
        linewidths=2
    )
    plt.clabel(CS, fmt=f'Recall={recall_threshold:.2f}', colors='red')

    # 保存图片
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {save_path}")
