import joblib
import os
import os.path as osp
import torch
import numpy as np

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
	classfied_data_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "2classfied_data.pkl")
	data_with_angles_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "3data_with_angles.pkl")

	classfied_data = joblib.load(classfied_data_path)

	new_data = []
	for sample in classfied_data:
		joints = sample["joints"]

		left_angles = []
		right_angles = []

		for joint_frame in joints:
			left_angle = compute_angle(joint_frame[11], joint_frame[13], joint_frame[11], joint_frame[4])
			right_angle = compute_angle(joint_frame[14], joint_frame[16], joint_frame[14], joint_frame[1])
			left_angles.append(left_angle)
			right_angles.append(right_angle)
		
		sample["left_arm_angles"] = left_angles
		sample["right_arm_angles"] = right_angles

		new_data.append(sample)
	
	joblib.dump(new_data, data_with_angles_path)
			
if __name__ == "__main__":
	gen_angles()
		
		