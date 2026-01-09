import os
import os.path as osp
import json
import joblib
import numpy as np

RAW_DATA_DIR = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")

def collect_data():
	label_data_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "1label_data.pkl")

	data = {}

	for huanz in os.listdir(RAW_DATA_DIR):
		kp_file = osp.join(RAW_DATA_DIR, huanz, "joints.json")
		label_file = osp.join(RAW_DATA_DIR, huanz, "descr.txt")

		with open(kp_file, "r") as f:
			content = json.load(f)

			xyz_data_per_frame = content["instance_info"]
			xyz_data_list = []

			for frame in xyz_data_per_frame:
				xyz_data_one_frame = frame["instances"][0]["keypoints"]
				xyz_data_list.append(xyz_data_one_frame)

			xyz_data = np.array(xyz_data_list)
		
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
			"joints": xyz_data,
			"total_label": total_label,
			"left_label": left_label,
			"right_label": right_label,
		}
		print(f"finish {huanz}")

	joblib.dump(data, label_data_path)

if __name__ == "__main__":
	collect_data()