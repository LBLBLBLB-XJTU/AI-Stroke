import joblib
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

CLASS0_PATH = "/home/liu_bang/AIStroke/data_generate/sample_txts/class_0.txt"
CLASS1_PATH = "/home/liu_bang/AIStroke/data_generate/sample_txts/class_1.txt"
CLASS2_PATH = "/home/liu_bang/AIStroke/data_generate/sample_txts/class_2.txt"

OUTPUT_DIR = "angle_vis"  # 可视化输出根目录

def add_sample_class():
    data_with_angles_path = "/home/liu_bang/AIStroke/data_generate/3data_with_angles.pkl"
    data = joblib.load(data_with_angles_path)

    # 读取每一类样本ID
    with open(CLASS0_PATH, "r") as f:
        class0_list = f.read().strip().split("\n")
    with open(CLASS1_PATH, "r") as f:
        class1_list = f.read().strip().split("\n")
    with open(CLASS2_PATH, "r") as f:
        class2_list = f.read().strip().split("\n")
    
    # 创建输出文件夹
    class_dirs = {}
    for i, cls_name in enumerate(["class0", "class1", "class2"]):
        dir_path = osp.join(OUTPUT_DIR, cls_name)
        os.makedirs(dir_path, exist_ok=True)
        class_dirs[i] = dir_path

    # 遍历每个样本
    for huanz in data:
        # 获取角度和id
        left_angles = np.array(huanz["left_arm_angles_0"])
        right_angles = np.array(huanz["right_arm_angles_0"])
        sample_id = huanz["id"]  # 这里 huanz 是键名，作为样本id使用

        # 判断类别
        if sample_id in class0_list:
            class_id = 0
        elif sample_id in class1_list:
            class_id = 1
        elif sample_id in class2_list:
            class_id = 2
        else:
            continue  # 不在任何类中则跳过

        # 计算平均值和差值
        left_mean = np.mean(left_angles)
        right_mean = np.mean(right_angles)
        L = min(len(left_angles), len(right_angles))
        diff = np.abs(left_angles[:L] - right_angles[:L])

        # 可视化
        plt.figure(figsize=(12,5))
        plt.plot(left_angles, label="Left Hand", marker='o')
        plt.plot(right_angles, label="Right Hand", marker='x')
        plt.plot(diff, label="Diff (abs(left-right))", color='gray', linestyle='--')

        # 左右手平均值画水平线
        plt.axhline(left_mean, color='blue', linestyle=':', label=f"Left Mean: {left_mean:.1f}")
        plt.axhline(right_mean, color='orange', linestyle=':', label=f"Right Mean: {right_mean:.1f}")

        plt.title(f"Sample: {sample_id} | Class: {class_id}")
        plt.xlabel("Frame Index")
        plt.ylabel("Angle (deg)")
        plt.legend()
        plt.tight_layout()

        # 保存到对应类别文件夹
        save_path = osp.join(class_dirs[class_id], f"{sample_id}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Finished {sample_id}.")

    print(f"角度可视化完成，保存在 {OUTPUT_DIR} 下的 class0/class1/class2 文件夹中。")

if __name__ == "__main__":
    add_sample_class()
