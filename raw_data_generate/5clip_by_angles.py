import joblib
import numpy as np
import os
import os.path as osp

DATA_W_ANGLES_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_label_data_with_angles.pkl")
CLIPPED_DATA_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_label_data_clipped.pkl")

import matplotlib.pyplot as plt

def plot_angle_with_segment(angle_list, segment, save_path=None, title="Angle Sequence"):
    angle_array = np.array(angle_list)

    plt.figure(figsize=(10, 4))
    plt.plot(angle_array, linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Angle (deg)")
    plt.title(title)

    if segment is not None:
        start, end = segment
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

def find_active_segment(
    angle_list,
    ratio_up_threshold=0.25,   # high ≥ low * 1.25
    ratio_down_threshold=0.20, # fall drop ≥ 20%
    plateau_min_frames=15,
    smooth=5
):
    import numpy as np
    
    angle = np.array(angle_list, dtype=float)

    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        angle_smooth = np.convolve(angle, kernel, mode="same")
    else:
        angle_smooth = angle.copy()

    peak = np.max(angle_smooth)
    plateau_tol = 0.2 * peak # 0.1

    peak_indices = np.where(angle_smooth >= peak - plateau_tol)[0]
    if len(peak_indices) == 0:
        return None

    # merge segments
    segments = []
    start = prev = None
    for idx in peak_indices:
        if start is None:
            start = idx
        elif idx != prev + 1:
            segments.append((start, prev))
            start = idx
        prev = idx
    segments.append((start, prev))

    valid_segments = [(s, e) for s, e in segments if (e - s + 1) >= plateau_min_frames]
    if not valid_segments:
        return None

    # pick longest
    s, e = max(valid_segments, key=lambda x: x[1]-x[0])
    plateau_mean = np.mean(angle_smooth[s:e+1])

    before = angle_smooth[:s]
    after  = angle_smooth[e+1:]

    cond_up = False
    cond_down = False

    # UP
    if len(before) > 0:
        valley_before = np.mean(before[-min(10, len(before)):])
        ratio_up = (plateau_mean - valley_before) / max(valley_before, 1e-3)
        cond_up = ratio_up >= ratio_up_threshold

    # DOWN
    if len(after) > 0:
        valley_after = np.mean(after[:min(10, len(after))])
        ratio_down = (plateau_mean - valley_after) / max(valley_after, 1e-3)
        cond_down = ratio_down >= ratio_down_threshold

    # CASES
    if cond_up and cond_down:
        return (s, e)
    if cond_up and len(after) == 0:
        return (s, e)
    if len(before) == 0 and cond_down:
        return (s, e)

    return None

def clip_samples_by_angles():
    data = joblib.load(DATA_W_ANGLES_PATH)
    new_clipped_data = []
    no_clipped = []

    for sample in data:
        if sample['id'] == "1180129448226328576":
            continue
        left_seg = find_active_segment(sample["left_arm_angles"])
        right_seg = find_active_segment(sample["right_arm_angles"])

        img_dir = osp.join("/home/liu_bang/angle_img", sample['id'])
        os.makedirs(img_dir, exist_ok=True)
        plot_angle_with_segment(sample["left_arm_angles"], left_seg, save_path=osp.join(img_dir, f"plot_left.png"), title=f"Left arm ({sample['id']})")
        plot_angle_with_segment(sample["right_arm_angles"], right_seg, save_path=osp.join(img_dir, f"plot_right.png"), title=f"Right arm ({sample['id']})")

        candidates = [seg for seg in [left_seg, right_seg] if seg is not None]
        if len(candidates) == 0:
            print(f"双侧都未检测到高区：{sample['id']}")
            no_clipped.append(sample['id'])
            clipped_sample = sample.copy()
            clipped_sample["clip_range"] = (0, -1, 0)
            clipped_sample["clip_len"] = 0
        else:
            start, end = min(candidates, key=lambda x: x[1] - x[0])

            # clip joints 与 angle
            clipped_sample = sample.copy()
            clipped_sample["joints_clipped"] = sample["joints"][start:end+1]
            clipped_sample["left_arm_angles_clipped"] = sample["left_arm_angles"][start:end+1]
            clipped_sample["right_arm_angles_clipped"] = sample["right_arm_angles"][start:end+1]

            # 保存抬手段长度
            seg_flag = 2
            if left_seg == None:
                seg_flag -= 1
            if right_seg == None:
                seg_flag -= 1
            clipped_sample["clip_range"] = (start, end, seg_flag)
            clipped_sample["clip_len"] = end - start + 1

        new_clipped_data.append(clipped_sample)

    joblib.dump(new_clipped_data, CLIPPED_DATA_PATH)
    print(f"Clipped samples saved: {len(new_clipped_data)}")
    print(no_clipped)
    
    CLASS0_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "sample_txts", "class_0.txt")
    def exclude_huanz(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.read().split()

        result = [line.strip() for line in lines if line.strip()]
        return result
    class0_huanz = exclude_huanz(CLASS0_PATH)
    for sample in new_clipped_data:
        if sample["clip_range"][2] != 2 and sample['id'] in class0_huanz:
            print(sample['id'])

if __name__ == "__main__":
    clip_samples_by_angles()
