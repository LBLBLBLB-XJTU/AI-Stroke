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

def stage1_rule_check():
    class_1_all = exclude_huanz(CLASS1_PATH)
    class_1_now = []
    # 这个才是能排除的，上面的排除了就数据泄露了
    classfied2wrong = []

    wrong_acc= 0
    data = joblib.load(CLIPPED_DATA_PATH)
    for sample in data:
        if sample['id'] in class_1_all:
            class_1_now.append(sample['id'])

        status = assess_sample(
            sample,
            min_angle_threshold=43,
            max_diff_threshold=12
        )
        
        if status[0] == "patient" and sample["total_label"] == 0:
            classfied2wrong.append((sample["id"], status[1]))
            wrong_acc += 1
        elif status[0] == "patient" and sample['total_label'] == 1:
            classfied2wrong.append((sample["id"], status[1]))

    class1_detected_id = []
    for sample in classfied2wrong:
        if sample[0] in class_1_now:
            class1_detected_id.append(sample[0])
    print(f"对于class1明显异常的判断检出率是{len(class1_detected_id) / len(class_1_now)}")
    missed_id = []
    for id in class_1_now:
        if id not in class1_detected_id:
            missed_id.append(id)
    print(f'未检测出的class1的id{missed_id}')

    print(f"对于所有判断成错误的样本，确实为错误的指标为{wrong_acc / len(classfied2wrong)}")
        
    ret_id = []
    for sample in classfied2wrong:
        ret_id.append(sample[0])
    return ret_id, class_1_now, []

def stage1_rule_check_test(test_idx, raw_data):
    class_1_all = exclude_huanz(CLASS1_PATH)
    class_1_idx = []

    indices = list(test_idx)

    idx_needed_net = []        # 需要送入网络的 idx（stage2）
    stage1_correct = 0         # stage1 判对的样本数

    for idx in indices:
        sample = raw_data[idx]

        status, reason = assess_sample(
            sample,
            min_angle_threshold=43,
            max_diff_threshold=12
        )

        gt = sample["total_label"]

        if status == "patient":
            # stage1 认为是明显异常
            if gt == 0:
                stage1_correct += 1   # 判对
            # 判错（gt=0）不加
        else:
            # stage1 放行，交给网络
            idx_needed_net.append(idx)
        
        if sample["id"] in class_1_all:
            class_1_idx.append(idx)

    return stage1_correct, idx_needed_net, class_1_idx