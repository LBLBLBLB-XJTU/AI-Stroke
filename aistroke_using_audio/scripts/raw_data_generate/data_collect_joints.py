import os
import os.path as osp
import joblib
import numpy as np
import torch
from smplx import SMPL

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

MESH_DIR = '/share/WHAM_Results_Lim'
EXCLUDE_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "exclude_huanz.txt")

def get_only_one_mesh(wham_output):
	merged_results = {}

	first_key = next(iter(wham_output))
	for k in wham_output[first_key].keys():
		merged_results[k] = []

	for _id, data in wham_output.items():
		for k, v in data.items():
			merged_results[k].append(v)

	for k, v in merged_results.items():
		try:
			merged_results[k] = np.stack(v)
		except:
			pass
	
	return merged_results

def exclude_huanz(txt_path):
	with open(txt_path, "r", encoding="utf-8") as f:
		lines = f.read().split()

	result = [line.strip() for line in lines if line.strip()]
	return result

def collect_whamfile():
	"""
	遍历所有任务，计算角度与左右手腕 XYZ 并保存
	"""
	num = 0
	model_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "smpl")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	smpl_model = SMPL(model_path, batch_size=1).to(device)

	results = {}  # ✅ 改为字典结构 {path: data}
	save_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "huanz_joints.pkl")
	exclude = exclude_huanz(EXCLUDE_PATH)

	for huanz in sorted(os.listdir(MESH_DIR)):
		if huanz in exclude:
			continue

		huanz_folder = osp.join(MESH_DIR, huanz)
		for task in sorted(os.listdir(huanz_folder)):
			if "LimbAction" in task:
				task_folder = osp.join(huanz_folder, task)
				wham_file = osp.join(task_folder, "wham_output.pkl")

				if osp.exists(wham_file):
					wham_output = joblib.load(wham_file)
					wham_output = get_only_one_mesh(wham_output)
					
					betas = torch.as_tensor(wham_output["betas"][0], dtype=torch.float32).to(device)            # (T,10)
					transl = torch.as_tensor(wham_output["trans_world"][0], dtype=torch.float32).to(device)       # (T,3)
					poses_72 = torch.as_tensor(wham_output["pose_world"][0], dtype=torch.float32).to(device)     # (T,72)
					# 拆分 pose 参数
					global_orient = poses_72[:, :3]              # (T,3) axis-angle
					body_pose = poses_72[:, 3:] 

					smploutput = smpl_model(betas=betas, global_orient=global_orient, body_pose=body_pose, transl=transl, return_verts=False, return_full_pose=False)

					results[huanz] = smploutput.joints  # ✅ 用路径作为键
					num += 1
					print(f"finish {wham_file}")

	# 保存为 joblib 文件
	joblib.dump(results, save_path)
	print(f"Saved raw whamfiles to {save_path} with {num} entries.")

if __name__ == "__main__":
	collect_whamfile()