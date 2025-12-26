import json
import joblib
from typing import Tuple
import os
import os.path as osp

LABEL_DIR = "/share/PatientVideo_Chang"
EXCLUDE_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "exclude_huanz.txt")

def check_txt(file_path: str) -> bool:
	"""检查txt文件是否非空"""
	return osp.exists(file_path) and osp.getsize(file_path) > 0

def load_label(label_path: str) -> Tuple[int, int, int]:
	"""
	解析 descr.txt，返回 (total_label, left_label, right_label)
	"""
	with open(label_path, "r", encoding="utf-8") as f:
		content = f.read().strip()
		gt_json = json.loads(content)

	# 默认正常为 1
	total_label = left_label = right_label = 1

	if "正常" not in gt_json.get("szjl", ""):
		total_label = 0
		left_label = 1
		right_label = 1
		if "异常" in gt_json.get("zszjlyc", ""):
			left_label = 0
		if "异常" in gt_json.get("yszjlyc", ""):
			right_label = 0
		assert left_label == 0 or right_label == 0, f"错误标签: {label_path}\n内容: {gt_json}"

	return total_label, left_label, right_label

def exclude_huanz(txt_path):
	with open(txt_path, "r", encoding="utf-8") as f:
		lines = f.read().split()

	result = [line.strip() for line in lines if line.strip()]
	return result

def collect_label():
	"""
	遍历所有任务，读取 descr.txt 并保存为 {path: label_dict} 结构
	"""
	results = {}
	save_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "label.pkl")
	num = 0
	exclude = exclude_huanz(EXCLUDE_PATH)

	for huanz in sorted(os.listdir(LABEL_DIR)):
		# 跳过异常样本
		if huanz in exclude:
			continue

		huanz_folder = osp.join(LABEL_DIR, huanz)
		for task in sorted(os.listdir(huanz_folder)):
			if "LimbAction" in task:
				task_folder = osp.join(huanz_folder, task)
				descr_path = osp.join(task_folder, "descr.txt")

				if check_txt(descr_path):
					total_label, left_label, right_label = load_label(descr_path)
					results[huanz] = {
						"total_label": total_label,
						"left_label": left_label,
						"right_label": right_label
					}
					num += 1
					print(f"Processed: {descr_path}")

	# 保存为 joblib 字典文件
	joblib.dump(results, save_path)
	print(f"Saved label dictionary with {num} entries to {save_path}")


if __name__ == "__main__":
	collect_label()
