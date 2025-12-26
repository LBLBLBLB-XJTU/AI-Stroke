import os
import os.path as osp
import joblib
import torch

RAW_LABEL_DATA_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_label_data.pkl")
DATA_W_ANGLES_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "raw_label_data_with_angles.pkl")

def compute_angles(start1, end1, start2, end2):
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
	return 180 - angle_deg.item()   # float

def get_arm_angles_one_frame():
	data = joblib.load(RAW_LABEL_DATA_PATH)
	new_data = []

	for sample in data:
		sample_id = sample["id"]
		sample_joints = sample["joints"]

		sample_left_arm_angles = []
		sample_right_arm_angles = []

		for joints_frame in sample_joints:
			left_arm_angle = compute_angles(joints_frame[16], joints_frame[18], joints_frame[1], joints_frame[16])
			right_arm_angle = compute_angles(joints_frame[17], joints_frame[19], joints_frame[2], joints_frame[17])
			
			sample_left_arm_angles.append(left_arm_angle)
			sample_right_arm_angles.append(right_arm_angle)
		
		sample["left_arm_angles"] = sample_left_arm_angles
		sample["right_arm_angles"] = sample_right_arm_angles

		new_data.append(sample)
		print(f"finish {sample_id}")

	joblib.dump(new_data, DATA_W_ANGLES_PATH)

if __name__ == "__main__":
	get_arm_angles_one_frame()