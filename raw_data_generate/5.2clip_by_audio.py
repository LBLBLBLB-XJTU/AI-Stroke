import joblib
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt

DATA_W_ANGLES_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_label_data_with_angles.pkl")
CLIPPED_DATA_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_label_data_clipped_byaudio.pkl")
AUDIO_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "Limb_Segment.txt")

def plot_angle_with_segment(angle_list, segment, save_path=None, title="Angle Sequence"):
    angle_array = np.array(angle_list)

    plt.figure(figsize=(10, 4))
    plt.plot(angle_array, linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Angle (deg)")
    plt.title(title)

    if segment is not None:
        start, end =  map(int, segment)
        plt.axvline(start, color='red', linestyle='--', linewidth=2, label=f"start={start}")
        plt.axvline(end, color='green', linestyle='--', linewidth=2, label=f"end={end}")
        plt.axhline(np.mean(angle_array[start:end+1]), color='orange', linestyle=':', label="segment avg")

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        # print(f"saved plot to {save_path}")
    else:
        plt.show()

    plt.close()

def clip_samples_by_angles():
    audio_info = []
    with open(AUDIO_PATH, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines:
        id = line.split("\t")[0]
        q1_start = float(line.split("\t")[1])
        q1_end = float(line.split("\t")[2])

        if q1_end > q1_start and q1_end != 0:
            audio_info.append({
                "id": id,
                "start": q1_start * 30 / 3,
                "end": q1_end * 30 / 3
            })

    data = joblib.load(DATA_W_ANGLES_PATH)
    clipped_data = []

    for sample in data:
        if sample['id'] == "1180129448226328576":
            continue
        for info in audio_info:
            if info["id"] == sample['id']:
                segment = (int(round(info["start"])), int(round(info["end"])))

                img_dir = osp.join("/home/liu_bang/angle_img_audio", sample['id'])
                os.makedirs(img_dir, exist_ok=True)
                plot_angle_with_segment(sample["left_arm_angles"], segment, save_path=osp.join(img_dir, f"plot_left.png"), title=f"Left arm ({sample['id']})")
                plot_angle_with_segment(sample["right_arm_angles"], segment, save_path=osp.join(img_dir, f"plot_right.png"), title=f"Right arm ({sample['id']})")
                start, end = segment

                # clip joints 与 angle
                clipped_sample = sample.copy()
                clipped_sample["joints"] = sample["joints"][start:end+1]

                clipped_sample["clip_range"] = (start, end)
                clipped_sample["clip_len"] = end - start + 1

                clipped_data.append(clipped_sample)

    joblib.dump(clipped_data, CLIPPED_DATA_PATH)
    print(f"Clipped samples saved: {len(clipped_data)}")

if __name__ == "__main__":
    clip_samples_by_angles()