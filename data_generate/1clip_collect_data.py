import os
import os.path as osp
import json
import numpy as np
import joblib

RAW_DATA_DIR = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
SEGMENT_FILE = osp.join(os.path.dirname(os.path.abspath(__file__)), "segment.txt")
LABEL_DATA_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "1label_data.pkl")

def parse_time_30fps(t):
    """
    '0130' -> total_frame (int)
    """
    sec = int(t[:2])
    frame = int(t[2:])
    return sec * 30 + frame

def load_segment_txt_30fps(txt_path):
    segments = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            video_id = parts[0]
            raw_times = parts[1:]

            frames = [parse_time_30fps(t) for t in raw_times]

            segments[video_id] = {
                "raw": raw_times,
                "frames": frames
            }

    return segments

def process_seg(segment, extend_sec, frame_len):
    if segment is None:
        return None

    segment_6points = []
    frames = segment["frames"]
    start1 = frames[0]
    start2 = frames[1]
    end2 = frames[2]
    end1 = frames[3]

    if start1 < 30 * extend_sec:
        start0 = 0
    else:
        start0 = start1 - 30 * extend_sec

    if end1 + 30 * extend_sec > frame_len:
        end0 = frame_len
    else:
        end0 = end1 + 30 * extend_sec

    segment_6points = [start0, start1, start2, end2, end1, end0]

    return segment_6points

def clip_collect_data():
    segments = load_segment_txt_30fps(SEGMENT_FILE)
    data = {}
    for huanz in os.listdir(RAW_DATA_DIR):
        kp_file = osp.join(RAW_DATA_DIR, huanz, "results.json")
        label_file = osp.join(RAW_DATA_DIR, huanz, "descr.txt")

        with open(kp_file, "r") as f:
            content = json.load(f)

            xyz_data_all_frame = content["instance_info"]
            
            # 前后扩展1S
            frame_len = len(xyz_data_all_frame)
            segment = process_seg(segments.get(huanz, None), 1, frame_len)

            xyz_data_list_0= []
            xyz_data_list_1 = []
            xyz_data_list_2 = []

            for frame in xyz_data_all_frame:
                frame_id = frame["frame_id"]
                if frame_id >= segment[0] and frame_id <= segment[5]:
                    xyz_data_list_0.append(frame["instances"][0]["keypoints"])
                if frame_id >= segment[1] and frame_id <= segment[4]:
                    xyz_data_list_1.append(frame["instances"][0]["keypoints"])
                if frame_id >= segment[2] and frame_id <= segment[3]:
                    xyz_data_list_2.append(frame["instances"][0]["keypoints"])
            
            xyz_data_0 = np.array(xyz_data_list_0)
            xyz_data_1 = np.array(xyz_data_list_1)
            xyz_data_2 = np.array(xyz_data_list_2)
        
        with open(label_file, "r") as f:
            content = f.read().strip()
            gt_json = json.loads(content)

            # 默认正常为 1
            total_label = left_label = right_label = 1

            if "正常" not in gt_json.get("szjl", ""):
                total_label = 0
                if "异常" in gt_json.get("zszjlyc", ""):
                    left_label = 0
                if "异常" in gt_json.get("yszjlyc", ""):
                    right_label = 0
                assert left_label == 0 or right_label == 0, f"错误标签: {huanz}\n内容: {gt_json}"

        data[huanz] = {
            "id": huanz,
            "joints_0": xyz_data_0,
            "joints_1": xyz_data_1,
            "joints_2": xyz_data_2,
            "total_label": total_label,
            "left_label": left_label,
            "right_label": right_label,
        }
        print(f"finish {huanz}")
    
    joblib.dump(data, LABEL_DATA_PATH)

if __name__ == "__main__":
    clip_collect_data()