import joblib
import os
import os.path as osp
import torch
import numpy as np

LABEL_DATA_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "1label_data.pkl")
DATA_WITH_ANGLES_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "2data_with_angles.pkl")

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

	return 180 - angle_deg.item()

def gen_angles():
	label_data = joblib.load(LABEL_DATA_PATH)

	new_data = []
	len_list0 = []
	len_list1 = []
	len_list2 = []
	pos_sample_count = 0
	neg_sample_count = 0

	for sample in label_data.values():
		joints_0 = sample["joints_0"]
		joints_1 = sample["joints_1"]
		joints_2 = sample["joints_2"]
	
		left_angles_0 = []
		right_angles_0 = []
		left_angles_1 = []
		right_angles_1 = []
		left_angles_2 = []
		right_angles_2 = []

		for joint_frame in joints_0:
			left_angles_0.append(
				compute_angle(joint_frame[11], joint_frame[13], joint_frame[11], joint_frame[4])
			)
			right_angles_0.append(
				compute_angle(joint_frame[14], joint_frame[16], joint_frame[14], joint_frame[1])
			)

		for joint_frame in joints_1:
			left_angles_1.append(
				compute_angle(joint_frame[11], joint_frame[13], joint_frame[11], joint_frame[4])
			)
			right_angles_1.append(
				compute_angle(joint_frame[14], joint_frame[16], joint_frame[14], joint_frame[1])
			)

		for joint_frame in joints_2:
			left_angles_2.append(
				compute_angle(joint_frame[11], joint_frame[13], joint_frame[11], joint_frame[4])
			)
			right_angles_2.append(
				compute_angle(joint_frame[14], joint_frame[16], joint_frame[14], joint_frame[1])
			)


		sample["left_arm_angles_0"] = left_angles_0
		sample["right_arm_angles_0"] = right_angles_0
		sample["left_arm_angles_1"] = left_angles_1
		sample["right_arm_angles_1"] = right_angles_1
		sample["left_arm_angles_2"] = left_angles_2
		sample["right_arm_angles_2"] = right_angles_2

		new_data.append(sample)
		len_list0.append(len(left_angles_0))
		len_list1.append(len(left_angles_1))
		len_list2.append(len(left_angles_2))
		if sample["total_label"] == 1:
			pos_sample_count += 1
		else:
			neg_sample_count += 1

		print(f"Processed sample {sample['id']}")
	
	avg_len0 = np.mean(len_list0)
	var_len0 = np.var(len_list0)
	avg_len1 = np.mean(len_list1)
	var_len1 = np.var(len_list1)
	avg_len2 = np.mean(len_list2)
	var_len2 = np.var(len_list2)
	joblib.dump(new_data, DATA_WITH_ANGLES_PATH)
	print(f"数据数量：{len(new_data)}")
	print(f"序列长度0平均值: {avg_len0:.2f}, 方差: {var_len0:.2f}, 最小值: {np.min(len_list0)}, 最大值: {np.max(len_list0)}")
	print(f"序列长度1平均值: {avg_len1:.2f}, 方差: {var_len1:.2f}, 最小值: {np.min(len_list1)}, 最大值: {np.max(len_list1)}")
	print(f"序列长度2平均值: {avg_len2:.2f}, 方差: {var_len2:.2f}, 最小值: {np.min(len_list2)}, 最大值: {np.max(len_list2)}")
			
if __name__ == "__main__":
	gen_angles()