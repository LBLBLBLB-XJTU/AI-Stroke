import joblib
import os
import os.path as osp
import numpy as np

CLASS0_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "sample_txts", "class_0.txt")
CLASS1_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "sample_txts", "class_1.txt")
CLASS2_PATH = osp.join(os.path.dirname(os.path.abspath(__file__)), "sample_txts", "class_2.txt")

def add_sample_class():
	label_data_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "1label_data.pkl")
	classfied_data_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "2classfied_data.pkl")

	data = joblib.load(label_data_path)

	with open(CLASS0_PATH, "r") as f:
		class0_list = f.read().strip().split("\n")
	with open(CLASS1_PATH, "r") as f:
		class1_list = f.read().strip().split("\n")
	with open(CLASS2_PATH, "r") as f:
		class2_list = f.read().strip().split("\n")

	classfied_data = []
	len_list = []
	for huanz in data.keys():
		len_list.append(data[huanz]["joints"].shape[0])
		if huanz in class0_list:
			classfied_data.append({
				"id": huanz,
				"joints": data[huanz]["joints"],
				"total_label": data[huanz]["total_label"],
				"left_label": data[huanz]["left_label"],
				"right_label": data[huanz]["right_label"],
				"sample_class": 0,
			})
		elif huanz in class1_list:
			classfied_data.append({
				"id": huanz,
				"joints": data[huanz]["joints"],
				"total_label": data[huanz]["total_label"],
				"left_label": data[huanz]["left_label"],
				"right_label": data[huanz]["right_label"],
				"sample_class": 1,
			})
		elif huanz in class2_list:
			classfied_data.append({
				"id": huanz,
				"joints": data[huanz]["joints"],
				"total_label": data[huanz]["total_label"],
				"left_label": data[huanz]["left_label"],
				"right_label": data[huanz]["right_label"],
				"sample_class": 2,
			})

	joblib.dump(classfied_data, classfied_data_path)

	avg_len = np.mean(len_list)
	var_len = np.var(len_list)

	print(f"有标签数据数量：{len(classfied_data)}")
	print(f"序列长度平均值: {avg_len:.2f}, 方差: {var_len:.2f}")

if __name__ == "__main__":
	add_sample_class()