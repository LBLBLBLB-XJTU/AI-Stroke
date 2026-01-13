import os
import os.path as osp
import sys
import time
import joblib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
import time

from config.config import parse_args
from utils.seed import set_random_seed
from utils.logger import setup_logger, log_scalars
from utils.choose_segment import choose_segment
from data.dataset import build_dataloaders
from model.MyNet import MyNet
from model.loss import Losses
from utils.engine import run_one_epoch
from stage1 import stage1_train, stage1_test

CLIPPED_DATA_PATH = osp.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_data_generate", "raw_label_data_clipped_byaudio.pkl")
# 项目路径
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# TODO
def main(): 
	cfg = parse_args()
	cfg.PROJECT_ROOT = PROJECT_ROOT
	set_random_seed(cfg.SEED_VALUE)

	device = torch.device(cfg.DEVICE)

	# 日志
	logs_root = os.path.join(cfg.PROJECT_ROOT, "logs")
	os.makedirs(logs_root, exist_ok=True)
	timestamp = time.strftime("%Y-%m-%d_%H%M%S")
	log_dir = os.path.join(logs_root, timestamp)
	os.makedirs(log_dir, exist_ok=True)
	logger = setup_logger(log_dir)
	logger.info(f"配置：{cfg}")

	# 加载数据
	raw_path = os.path.join(os.path.dirname(cfg.PROJECT_ROOT), cfg.PATH.DATA_PATH)
	raw_data = joblib.load(raw_path)
	raw_data = choose_segment(raw_data, cfg)
	labels = [d["total_label"] for d in raw_data]

	# 外层 fold: 测试
	skf_outer = StratifiedKFold(n_splits=cfg.OUTER_FOLDS, shuffle=True, random_state=cfg.SEED_VALUE)
	outer_fold_acc = []
	outer_fold_net_acc = []

	for outer_fold, (outer_trainval_idx, outer_test_idx) in enumerate(skf_outer.split(np.zeros(len(labels)), labels)):
		logger.info(f"===== Outer Fold {outer_fold} =====")
		outer_fold_dir = os.path.join(log_dir, f"fold{outer_fold}")
		os.makedirs(outer_fold_dir, exist_ok=True)

		best_params = stage1_train(outer_trainval_idx, raw_data, cfg)
		logger.info(f"stage1参数选择：{best_params}")

		outer_trainval_idx = list(outer_trainval_idx)
		outer_test_idx = list(outer_test_idx)

		# 内层 fold: train/val
		skf_inner = StratifiedKFold(n_splits=cfg.INNER_FOLDS, shuffle=True, random_state=cfg.SEED_VALUE)
		inner_best_val_acc = -1.0  # ⭐外层记录：最好的内层模型 acc
		inner_best_model = None 

		for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(skf_inner.split(
			np.zeros(len(outer_trainval_idx)),
			[labels[i] for i in outer_trainval_idx]
		)):
			logger.info(f"--- Inner Fold {inner_fold} ---")
			inner_fold_dir = os.path.join(outer_fold_dir, f"inner_fold{inner_fold}")
			os.makedirs(inner_fold_dir, exist_ok=True)

			inner_train_idx = [outer_trainval_idx[i] for i in inner_train_idx]
			inner_val_idx = [outer_trainval_idx[i] for i in inner_val_idx]

			train_loader, val_loader, _ = build_dataloaders(
				cfg,
				train_idx=inner_train_idx,
				val_idx=inner_val_idx,
				test_idx=[],
				raw_data=raw_data,
				device=device
			)
	
			net = MyNet(cfg).to(device)
			optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.T_MAX , eta_min=1e-6)
			criterion = Losses(cfg)
			lr_max = cfg.TRAIN.LR
			lr_min = 1e-6

			logger.info(f"模型结构：{net}")
			logger.info(f"optimizer: {optimizer}")
			logger.info(f"scheduler: {scheduler}")
			logger.info(f"criterion: {criterion}")

			writer = SummaryWriter(log_dir=inner_fold_dir)

			inner_fold_best_acc = -1.0  # ⭐本折的最佳 acc
			ckpt_path = os.path.join(inner_fold_dir, f"inner_best_model_fold{inner_fold}.pth")

			for epoch in range(cfg.TRAIN.EPOCH):
				if epoch < cfg.TRAIN.WARMUP_EPOCHS:
					lr = lr_min + (lr_max - lr_min) * (epoch + 1) / cfg.TRAIN.WARMUP_EPOCHS
					for param_group in optimizer.param_groups:
						param_group['lr'] = lr

				train_results = run_one_epoch(net, train_loader, device, criterion, optimizer, mode="train")
				val_results = run_one_epoch(net, val_loader, device, criterion, mode="eval")

				log_scalars(writer, logger, "Train", train_results, epoch)
				log_scalars(writer, logger, "Valid", val_results, epoch)

				# 保存最优模型
				if val_results["final_acc"] > inner_fold_best_acc + 1e-6:
					inner_fold_best_acc = val_results["final_acc"]
					torch.save({
						"epoch": epoch,
						"model_state_dict": net.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
						"val_acc": inner_fold_best_acc,
						"outer_fold": outer_fold,
						"inner_fold": inner_fold
					}, ckpt_path)
					logger.info(f"✅ 保存最佳模型: {ckpt_path}")
				scheduler.step()

			writer.close()

			# ⭐ 比较本内层折与外层最佳
			if inner_fold_best_acc > inner_best_val_acc + 1e-6:
				inner_best_val_acc = inner_fold_best_acc
				inner_best_model = ckpt_path
				logger.info(f"🔥 更新外层最佳内层模型：{inner_best_model}，acc={inner_best_val_acc:.4f}")

		# 外层测试
		if cfg.USE_STAGE1:
			stage1_correct, idx_needed_net= stage1_test(outer_test_idx, raw_data, best_params)
		else:
			stage1_correct = 0
			idx_needed_net = outer_test_idx
		train_loader, val_loader, test_loader = build_dataloaders(
				cfg,
				train_idx=[],
				val_idx=[],
				test_idx=idx_needed_net,
				raw_data=raw_data,
				device=device
			)
		
		# 加载内层最优模型
		checkpoint = torch.load(inner_best_model, map_location=device)
		net.load_state_dict(checkpoint["model_state_dict"])
		criterion = Losses(cfg)

		test_results = run_one_epoch(net, test_loader, device, criterion, mode="test", save_dir=outer_fold_dir)
		stage2_correct = test_results["stage2_correct"]

		final_acc = (stage1_correct + stage2_correct) / len(outer_test_idx)
		net_acc = test_results["final_acc"]
		
		logger.info(f"Outer Fold {outer_fold} 的全系统正确率为:")
		logger.info(f"{final_acc}")
		logger.info(f"Outer Fold {outer_fold} 的网络正确率为")
		logger.info(f"{net_acc}")
		outer_fold_acc.append(final_acc)
		outer_fold_net_acc.append(net_acc)

	avg_acc = sum(outer_fold_acc) / len(outer_fold_acc)
	avg_net_acc = sum(outer_fold_net_acc) / len(outer_fold_net_acc)
	logger.info(f"所有折的测试结果{outer_fold_acc} =====")
	logger.info(f"所有折的网络测试结果{outer_fold_net_acc} =====")
	logger.info(f"===== 双重交叉平均最终测试准确率: {avg_acc:.4f} =====")
	logger.info(f"===== 双重交叉平均最终测试网络准确率: {avg_net_acc:.4f} =====")

if __name__ == "__main__":
	main()