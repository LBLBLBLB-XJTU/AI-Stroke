import joblib
import os
import os.path as osp
import numpy as np

CLASS0_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "sample_txts", "class_0.txt")
CLASS1_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "sample_txts", "class_1.txt")
CLASS2_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "sample_txts", "class_2.txt")

def exclude_huanz(txt_path):
	with open(txt_path, "r", encoding="utf-8") as f:
		lines = f.read().split()

	result = [line.strip() for line in lines if line.strip()]
	return result

def collect_data() -> None:
    joints_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "huanz_joints.pkl")
    labels_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "label.pkl")

    raw_label_data_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_label_data.pkl")

    label_data = []
    joints = joblib.load(joints_path)
    labels = joblib.load(labels_path)
    len_list = []

    class0_huanz = exclude_huanz(CLASS0_PATH)
    class1_huanz = exclude_huanz(CLASS1_PATH)
    class2_huanz = exclude_huanz(CLASS2_PATH)

    for huanz in sorted(joints.keys()):
        joint = joints[huanz][:, :24]
        assert joint.shape[1] == 24

        if huanz in labels.keys():
            if huanz == '1180129448226328576':
                continue
            len_list.append(joint.shape[0])
            if huanz in class0_huanz:
                sample_class = 0
                label_data.append({
                "id": huanz,
                "joints": joint,
                "left_label": labels[huanz]["left_label"],
                "right_label": labels[huanz]["right_label"],
                "total_label": labels[huanz]["total_label"],
                "sample_class": sample_class
            })
            elif huanz in class1_huanz:
                sample_class = 1
                label_data.append({
                "id": huanz,
                "joints": joint,
                "left_label": labels[huanz]["left_label"],
                "right_label": labels[huanz]["right_label"],
                "total_label": labels[huanz]["total_label"],
                "sample_class": sample_class
            })
            elif huanz in class2_huanz:
                sample_class = 2
                label_data.append({
                "id": huanz,
                "joints": joint,
                "left_label": labels[huanz]["left_label"],
                "right_label": labels[huanz]["right_label"],
                "total_label": labels[huanz]["total_label"],
                "sample_class": sample_class
            })

    # 保存数据
    os.makedirs(os.path.dirname(raw_label_data_path), exist_ok=True)
    joblib.dump(label_data, raw_label_data_path)

    # 计算平均长度和方差
    avg_len = np.mean(len_list)
    var_len = np.var(len_list)

    print(f"有标签数据数量：{len(label_data)}")
    print(f"序列长度平均值: {avg_len:.2f}, 方差: {var_len:.2f}")

if __name__ == "__main__":
    collect_data()
