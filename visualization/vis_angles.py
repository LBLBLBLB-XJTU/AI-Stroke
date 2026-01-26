import os
import os.path as osp
import joblib
import numpy as np
import matplotlib.pyplot as plt

def vis_trspd_angles():
    TRSPD_DATA_PATH = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "data", "TRSPD", "trspd_data.pkl")
    IMG_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "trspd_imgs")
    os.makedirs(IMG_DIR, exist_ok=True)
    data = joblib.load(TRSPD_DATA_PATH)

    for idx, sample in enumerate(data):
        left_angles = np.array(sample["left_angles"])
        right_angles = np.array(sample["right_angles"])
        label = sample["total_label"]

        label_name = "Healthy" if label == 1 else "Patient"

        plt.figure(figsize=(10, 4))
        plt.plot(left_angles, label="Left Arm", color="tab:blue")
        plt.plot(right_angles, label="Right Arm", color="tab:orange")

        plt.xlabel("Frame")
        plt.ylabel("Arm Raise Angle (deg)")
        plt.title(f"{label_name} Sample #{idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(osp.join((IMG_DIR),f"sample_{idx}"))

def vis_mydata_angles():
    MYDATA_PATH = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "data", "MYDATA", "my_data.pkl")
    IMG_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "mydata_imgs")
    os.makedirs(IMG_DIR, exist_ok=True)
    data = joblib.load(MYDATA_PATH)

    segment_pairs = {(0, 5): "red", (1, 4): "green", (2, 3): "blue"}

    for idx, sample in enumerate(data):
        huanz_id = sample["id"]
        left_angles_all = np.array(sample["left_arm_angles_all"])
        right_angles_all = np.array(sample["right_arm_angles_all"])
        segments = sample["segment"]

        T = len(left_angles_all)
        x = np.arange(T)

        plt.figure(figsize=(12, 5))

        # ===== 1️⃣ 画左右角度 =====
        plt.plot(x, left_angles_all, label="Left Arm Angle", color="orange", linewidth=2)
        plt.plot(x, right_angles_all, label="Right Arm Angle", color="purple", linewidth=2)

        # ===== 2️⃣ 画 segment 垂直线 =====
        for (i, j), color in segment_pairs.items():
            if i < len(segments) and j < len(segments):
                xi = segments[i]
                xj = segments[j]

                plt.axvline(x=xi, color=color, linestyle="--", alpha=0.7)
                plt.axvline(x=xj, color=color, linestyle="--", alpha=0.7)

                # 可选：在上方标注
                plt.text(xi, plt.ylim()[1], f"{i}",
                         color=color, ha="center", va="bottom", fontsize=9)
                plt.text(xj, plt.ylim()[1], f"{j}",
                         color=color, ha="center", va="bottom", fontsize=9)

        # ===== 3️⃣ 图像装饰 =====
        plt.title(f"Sample {idx} Arm Angles with Segments")
        plt.xlabel("Frame Index")
        plt.ylabel("Angle Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(osp.join(IMG_DIR, f"{huanz_id}.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    vis_trspd_angles()
    vis_mydata_angles()
