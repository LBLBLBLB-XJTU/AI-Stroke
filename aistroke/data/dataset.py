import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from .data_augment import joints_augment
from .feat_generate import generate_feat
from pytorch_wavelets import DWT1DForward

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
		self.use_aug = (mode == "train")
		self.modalities_used = [cfg.MODEL.MODALITIES_NAMES[i] for i in cfg.MODEL.MODALITIES_USED_IDX]

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		sample = self.data[idx]
		
		huanz_id = sample["id"]
		if self.cfg.MODEL.USE_CLIPPED:
			if sample["clip_range"][2] == 0:
				joints = sample["joints"]
			else:
				joints = sample["joints_clipped"]
		else:
			joints = sample["joints"]

		joints_aug = joints_augment(joints, self.cfg)
		feats = generate_feat(self.cfg, joints_aug, self.device, self.modalities_used)

		left_label = torch.tensor(sample["left_label"], dtype=torch.long)
		right_label = torch.tensor(sample["right_label"], dtype=torch.long)
		total_label = torch.tensor(sample["total_label"], dtype=torch.long)
		return huanz_id, left_label, right_label, total_label, feats
	  
class BucketBatchSampler(Sampler):
	def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
		self.dataset = dataset
		self.batch_size = batch_size
		self.drop_last = drop_last
		self.shuffle = shuffle
		# 用左臂角度序列长度做排序示例
		self.lengths = [d[4]["joints"].shape[0] for d in dataset]
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
	batch_size = len(batch)
	sample_results = batch[0][4]
	modalities = list(sample_results.keys())

	max_len = max(b[4]["joints"].shape[0] for b in batch)

	batch_tensors = {}
	for m in modalities:
		s_shape = sample_results[m].shape[1:]
		batch_shape = (batch_size, max_len, *s_shape)
		batch_tensors[m] = torch.zeros(batch_shape, dtype=torch.float32)

	left_labels, right_labels, all_labels = [], [], []
	huanz_ids = []

	for i, s in enumerate(batch):
		seq_len = s[4]["joints"].shape[0]
		for m in modalities:
			batch_tensors[m][i, :seq_len, ...] = s[4][m]

		huanz_ids.append(s[0])
		left_labels.append(s[1])
		right_labels.append(s[2])
		all_labels.append(s[3])

	return (
		huanz_ids,
		torch.tensor(left_labels, dtype=torch.long),
		torch.tensor(right_labels, dtype=torch.long),
		torch.tensor(all_labels, dtype=torch.long),
		batch_tensors
	)

def build_dataloaders(cfg, train_idx, val_idx, test_idx, raw_data, device):
	train_dataset = LabelAngleDataset(cfg, "train", train_idx, raw_data, device)
	val_dataset = LabelAngleDataset(cfg, "val", val_idx, raw_data, device)
	test_dataset = LabelAngleDataset(cfg, "test", test_idx, raw_data, device)

	train_sampler = BucketBatchSampler(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
	val_sampler = BucketBatchSampler(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)
	test_sampler = BucketBatchSampler(test_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

	train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=label_collate_fn, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=label_collate_fn, pin_memory=True)
	test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=label_collate_fn, pin_memory=True)

	return train_loader, val_loader, test_loader