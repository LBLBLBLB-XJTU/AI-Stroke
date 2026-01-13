import joblib
import os
import os.path as osp
import torch
import numpy as np

CLASSFIED_DATA_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "2classfied_data.pkl")
DATA_WITH_ANGLES_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "3data_with_angles.pkl")

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
	classfied_data = joblib.load(CLASSFIED_DATA_PATH)

	new_data = []
	for sample in classfied_data:
		joints_0 = sample["joints_0"]
		joints_1 = sample["joints_1"]
		joints_2 = sample["joints_2"]
	
		left_angles_0 = []
		right_angles_0 = []
		left_angles_1 = []
		right_angles_1 = []
		left_angles_2 = []
		right_angles_2 = []

		for joint_frame_0, joint_frame_1, joint_frame_2 in zip(joints_0, joints_1, joints_2):
			left_angle_0 = compute_angle(joint_frame_0[11], joint_frame_0[13], joint_frame_0[11], joint_frame_0[4])
			right_angle_0 = compute_angle(joint_frame_1[14], joint_frame_1[16], joint_frame_1[14], joint_frame_1[1])
			left_angles_0.append(left_angle_0)
			right_angles_0.append(right_angle_0)

			left_angle_1 = compute_angle(joint_frame_1[11], joint_frame_1[13], joint_frame_1[11], joint_frame_1[4])
			right_angle_1 = compute_angle(joint_frame_1[14], joint_frame_1[16], joint_frame_1[14], joint_frame_1[1])
			left_angles_1.append(left_angle_1)
			right_angles_1.append(right_angle_1)

			left_angle_2 = compute_angle(joint_frame_2[11], joint_frame_2[13], joint_frame_2[11], joint_frame_2[4])
			right_angle_2 = compute_angle(joint_frame_2[14], joint_frame_2[16], joint_frame_2[14], joint_frame_2[1])
			left_angles_2.append(left_angle_2)
			right_angles_2.append(right_angle_2)


		sample["left_arm_angles_0"] = left_angles_0
		sample["right_arm_angles_0"] = right_angles_0
		sample["left_arm_angles_1"] = left_angles_1
		sample["right_arm_angles_1"] = right_angles_1
		sample["left_arm_angles_2"] = left_angles_2
		sample["right_arm_angles_2"] = right_angles_2
		new_data.append(sample)
		print(f"Processed sample {sample['id']}")
	
	joblib.dump(new_data, DATA_WITH_ANGLES_PATH)
			
if __name__ == "__main__":
	gen_angles()
		
		