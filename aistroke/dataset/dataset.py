import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

from .feat_generate import generate_feat

class LabelAngleDataset(Dataset):
	def __init__(self, cfg, mode, indices, raw_data, device):
		"""
		raw_data: 必须传入同一份数据，避免重复加载
		indices: list[int], 空列表也可以
		"""
		self.cfg = cfg
		self.mode = mode
		self.device = device

		self.label_data = raw_data
		self.indices = list(indices)  # 必须是 list，即使为空
		self.data = [self.label_data[i] for i in self.indices]

		# train 阶段是否使用增强
		self.modalities_used = [cfg.MODEL.MODALITIES_NAMES[i] for i in cfg.MODEL.MODALITIES_USED_IDX]

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		sample = self.data[idx]
		
		huanz_id = sample["id"]
		joints = sample["joints"]

		feats = generate_feat(self.cfg, joints, self.device, self.modalities_used)

		total_label = torch.tensor(sample["total_label"], dtype=torch.long)
		if "left_label" in sample.keys():
			left_label = torch.tensor(sample["left_label"], dtype=torch.long)
		else:
			left_label = None
		if "right_label" in sample.keys():
			right_label = torch.tensor(sample["right_label"], dtype=torch.long)
		else:
			right_label = None
			
		return {"huanz_id": huanz_id,
		  "left_label": left_label,
		  "right_label": right_label,
		  "total_label": total_label,
		  "modals": feats}
	  
class BucketBatchSampler(Sampler):
	def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
		self.dataset = dataset
		self.batch_size = batch_size
		self.drop_last = drop_last
		self.shuffle = shuffle

		first_sample = dataset[0]["modals"]
		self.first_modality = next(iter(first_sample.keys()))
		# 用第一个输入模态获取长度
		self.lengths = [d["modals"][self.first_modality].shape[0] for d in dataset]
		self.bins = np.argsort(self.lengths)

	def __iter__(self):
		bins = self.bins.copy()
		if self.shuffle:
			np.random.shuffle(bins)

		batch = []
		for idx in bins:
			batch.append(idx)
			if len(batch) == self.batch_size:
				yield batch
				batch = []

		if len(batch) > 0 and not self.drop_last:
			yield batch

	def __len__(self):
		if self.drop_last:
			return len(self.dataset) // self.batch_size
		else:
			return (len(self.dataset) + self.batch_size - 1) // self.batch_size
		
def label_collate_fn(batch):
	IGNORE_LABEL = -1

	batch_size = len(batch)
	sample_results = batch[0]["modals"]
	modalities = list(sample_results.keys())

	max_len = max(b["modals"][modalities[0]].shape[0] for b in batch)

	batch_tensors = {}
	for m in modalities:
		s_shape = sample_results[m].shape[1:]
		batch_shape = (batch_size, max_len, *s_shape)
		batch_tensors[m] = torch.zeros(batch_shape, dtype=torch.float32)

	huanz_ids, left_labels, right_labels, total_labels = [], [], [], []

	for i, s in enumerate(batch):
		seq_len = s["modals"][modalities[0]].shape[0]
		for m in modalities:
			batch_tensors[m][i, :seq_len, ...] = s["modals"][m]

		huanz_ids.append(s["huanz_id"])
		left_labels.append(s["left_label"] if s["left_label"] is not None else IGNORE_LABEL)
		right_labels.append(s["right_label"] if s["right_label"] is not None else IGNORE_LABEL)
		total_labels.append(s["total_label"])

	return {"huanz_ids": huanz_ids,
		 "left_labels": torch.tensor(left_labels, dtype=torch.long),
		 "right_labels": torch.tensor(right_labels, dtype=torch.long),
		 "total_labels": torch.tensor(total_labels, dtype=torch.long),
		 "inputs": batch_tensors
		 }

def build_dataloaders(cfg, train_idx, val_idx, test_idx, raw_data, device):
	train_loader = None
	val_loader = None
	test_loader = None
		
	if len(train_idx) > 0:
		train_dataset = LabelAngleDataset(cfg, "train", train_idx, raw_data, device)
		train_sampler = BucketBatchSampler(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
		train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=label_collate_fn, pin_memory=True)
	if len(val_idx) > 0:
		val_dataset = LabelAngleDataset(cfg, "val", val_idx, raw_data, device)
		val_sampler = BucketBatchSampler(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
		val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=label_collate_fn, pin_memory=True)
	if len(test_idx) > 0:
		test_dataset = LabelAngleDataset(cfg, "test", test_idx, raw_data, device)
		test_sampler = BucketBatchSampler(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
		test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=label_collate_fn, pin_memory=True)

	return train_loader, val_loader, test_loader