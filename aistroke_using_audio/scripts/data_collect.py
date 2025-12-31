import joblib
import os
import os.path as osp
import numpy as np

def collect_data() -> None:
    joints_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_data_generate", "huanz_joints.pkl")
    labels_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_data_generate", "label.pkl")

    raw_label_data_path = osp.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/raw_data/raw_label_data.pkl")

    label_data = []
    joints = joblib.load(joints_path)
    labels = joblib.load(labels_path)
    len_list = []

    for huanz in sorted(joints.keys()):
        joint = joints[huanz][:, :24]
        assert joint.shape[1] == 24

        if huanz in labels.keys():
            len_list.append(joint.shape[0])
            label_data.append({
                "id": huanz,
                "joints": joint,
                "left_label": labels[huanz]["left_label"],
                "right_label": labels[huanz]["right_label"],
                "total_label": labels[huanz]["total_label"]
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
