# TODO
import os
import os.path as osp
import json
import numpy as np
import joblib
import torch

DATA_ROOT_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "data", "MYDATA")
RAW_DATA_DIR = osp.join(DATA_ROOT_DIR, "raw_data")
SEGMENT_FILE_PATH = osp.join(DATA_ROOT_DIR, "segment.txt")
DATA_WITH_ANGLES_PATH = osp.join(DATA_ROOT_DIR, "my_data.pkl")

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

def clip_collect_data(UPPER_MASK):
	segments = load_segment_txt_30fps(SEGMENT_FILE_PATH)
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

			xyz_data_list_all = []
			xyz_data_list_0 = []
			xyz_data_list_1 = []
			xyz_data_list_2 = []

			for frame in xyz_data_all_frame:
				frame_id = frame["frame_id"]
				xyz_data_list_all.append(frame["instances"][0]["keypoints"])
				if frame_id >= segment[0] and frame_id <= segment[5]:
					xyz_data_list_0.append(frame["instances"][0]["keypoints"])
				if frame_id >= segment[1] and frame_id <= segment[4]:
					xyz_data_list_1.append(frame["instances"][0]["keypoints"])
				if frame_id >= segment[2] and frame_id <= segment[3]:
					xyz_data_list_2.append(frame["instances"][0]["keypoints"])
			
			xyz_data_all = np.array(xyz_data_list_all)[:,UPPER_MASK,:]
			xyz_data_0 = np.array(xyz_data_list_0)[:,UPPER_MASK,:]
			xyz_data_1 = np.array(xyz_data_list_1)[:,UPPER_MASK,:]
			xyz_data_2 = np.array(xyz_data_list_2)[:,UPPER_MASK,:]
		
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
			"joints_all": xyz_data_all,
			"joints_0": xyz_data_0,
			"joints_1": xyz_data_1,
			"joints_2": xyz_data_2,
			"total_label": total_label,
			"left_label": left_label,
			"right_label": right_label,
			"segment": segment
		}
		print(f"finish {huanz}")
	
	return data

def compute_angle(start1, end1, start2, end2):
	start1 = torch.tensor(start1, dtype=torch.float32) if isinstance(start1, np.ndarray) else start1
	end1 = torch.tensor(end1, dtype=torch.float32) if isinstance(end1, np.ndarray) else end1
	start2 = torch.tensor(start2, dtype=torch.float32) if isinstance(start2, np.ndarray) else start2
	end2 = torch.tensor(end2, dtype=torch.float32) if isinstance(end2, np.ndarray) else end2
	
	vec1 = start1 - end1
	vec2 = start2 - end2

	norm1 = torch.norm(vec1)
	norm2 = torch.norm(vec2)

	if norm1 < 1e-6 or norm2 < 1e-6:
		return 0.0

	vec1_unit = vec1 / norm1
	vec2_unit = vec2 / norm2
	cos_theta = torch.clamp(torch.dot(vec1_unit, vec2_unit), -1.0, 1.0)

	angle_rad = torch.acos(cos_theta)
	angle_deg = torch.rad2deg(angle_rad)

	return angle_deg.item()

def gen_angles(label_data):
	new_data = []
	len_list_all = []
	len_list0 = []
	len_list1 = []
	len_list2 = []
	pos_sample_count = 0
	neg_sample_count = 0

	for sample in label_data.values():
		joints_all = sample["joints_all"]
		joints_0 = sample["joints_0"]
		joints_1 = sample["joints_1"]
		joints_2 = sample["joints_2"]

		left_angles_all = []
		right_angles_all = []
		left_angles_0 = []
		right_angles_0 = []
		left_angles_1 = []
		right_angles_1 = []
		left_angles_2 = []
		right_angles_2 = []

		for joint_frame in joints_all:
			left_angles_all.append(
				compute_angle(joint_frame[0], joint_frame[3], joint_frame[5], joint_frame[7])
			)
			right_angles_all.append(
				compute_angle(joint_frame[0], joint_frame[3], joint_frame[8], joint_frame[10])
			)

		for joint_frame in joints_0:
			left_angles_0.append(
				compute_angle(joint_frame[0], joint_frame[3], joint_frame[5], joint_frame[7])
			)
			right_angles_0.append(
				compute_angle(joint_frame[0], joint_frame[3], joint_frame[8], joint_frame[10])
			)

		for joint_frame in joints_1:
			left_angles_1.append(
				compute_angle(joint_frame[0], joint_frame[3], joint_frame[5], joint_frame[7])
			)
			right_angles_1.append(
				compute_angle(joint_frame[0], joint_frame[3], joint_frame[8], joint_frame[10])
			)

		for joint_frame in joints_2:
			left_angles_2.append(
				compute_angle(joint_frame[0], joint_frame[3], joint_frame[5], joint_frame[7])
			)
			right_angles_2.append(
				compute_angle(joint_frame[0], joint_frame[3], joint_frame[8], joint_frame[10])
			)

		sample["left_arm_angles_all"] = left_angles_all
		sample["right_arm_angles_all"] = right_angles_all
		sample["left_arm_angles_0"] = left_angles_0
		sample["right_arm_angles_0"] = right_angles_0
		sample["left_arm_angles_1"] = left_angles_1
		sample["right_arm_angles_1"] = right_angles_1
		sample["left_arm_angles_2"] = left_angles_2
		sample["right_arm_angles_2"] = right_angles_2

		new_data.append(sample)
		len_list_all.append(len(left_angles_all))
		len_list0.append(len(left_angles_0))
		len_list1.append(len(left_angles_1))
		len_list2.append(len(left_angles_2))
		if sample["total_label"] == 1:
			pos_sample_count += 1
		else:
			neg_sample_count += 1

		print(f"Processed sample {sample['id']}")
	
	avg_len_all = np.mean(len_list_all)
	var_len_all = np.var(len_list_all)
	avg_len0 = np.mean(len_list0)
	var_len0 = np.var(len_list0)
	avg_len1 = np.mean(len_list1)
	var_len1 = np.var(len_list1)
	avg_len2 = np.mean(len_list2)
	var_len2 = np.var(len_list2)
	joblib.dump(new_data, DATA_WITH_ANGLES_PATH)
	print(f"数据数量：{len(new_data)}")
	print(f"完整序列平均值: {avg_len_all:.2f}, 方差: {var_len_all:.2f}, 最小值: {np.min(len_list_all)}, 最大值: {np.max(len_list_all)}")
	print(f"序列长度0平均值: {avg_len0:.2f}, 方差: {var_len0:.2f}, 最小值: {np.min(len_list0)}, 最大值: {np.max(len_list0)}")
	print(f"序列长度1平均值: {avg_len1:.2f}, 方差: {var_len1:.2f}, 最小值: {np.min(len_list1)}, 最大值: {np.max(len_list1)}")
	print(f"序列长度2平均值: {avg_len2:.2f}, 方差: {var_len2:.2f}, 最小值: {np.min(len_list2)}, 最大值: {np.max(len_list2)}")
	 
if __name__ == "__main__":
	UPPER_MASK = [0,7,8,9,10,11,12,13,14,15,16]
	label_data = clip_collect_data(UPPER_MASK)
	gen_angles(label_data)
