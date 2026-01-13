import os
import sys
import time
import joblib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from collections import Counter

from config.config import parse_args
from utils.seed import set_random_seed
from utils.logger import setup_logger, log_scalars
from utils.choose_segment import choose_segment
from data.dataset import build_dataloaders
from model.MyNet import MyNet
from model.loss import Losses
from utils.engine import run_one_epoch_cosface
from stage1 import stage1_test, stage1_train

# 项目路径
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

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

    # 简单划分 train/val/test (比如 70%/15%/15%)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)), test_size=cfg.SPLIT_RATIO[2] / sum(cfg.SPLIT_RATIO), stratify=labels, random_state=cfg.SEED_VALUE
    )
    best_params = stage1_train(train_val_idx, raw_data, cfg)
    logger.info(f"stage1参数选择：{best_params}")
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=cfg.SPLIT_RATIO[1] / (cfg.SPLIT_RATIO[0] + cfg.SPLIT_RATIO[1]),
        stratify=[labels[i] for i in train_val_idx],
        random_state=cfg.SEED_VALUE
    )

    logger.info(f"训练集样本数: {len(train_idx)}, 验证集样本数: {len(val_idx)}, 测试集样本数: {len(test_idx)}")
    test_labels = [labels[i] for i in test_idx]
    counter = Counter(test_labels)
    logger.info("测试集类别分布：")
    for k, v in counter.items():
        logger.info(f"  label={k}: {v} 个样本")

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg,
        train_idx=train_idx,
        # train_idx=np.concatenate([train_idx, val_idx, test_idx]),
        val_idx=val_idx,
        # val_idx=np.concatenate([train_idx, val_idx, test_idx]),
        test_idx=[],
        raw_data=raw_data,
        device=device
    )

    # 模型、优化器、损失
    net = MyNet(cfg).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.T_MAX, eta_min=1e-6)
    criterion = Losses(cfg)
    warmup_epochs = 15
    lr_max = cfg.TRAIN.LR
    lr_min = 1e-6
    
    logger.info(f"模型结构：{net}")
    logger.info(f"optimizer: {optimizer}")
    logger.info(f"scheduler: {scheduler}")
    logger.info(f"criterion: {criterion}")

    writer = SummaryWriter(log_dir=log_dir)
    best_acc = 0.0
    ckpt_path = os.path.join(log_dir, "best_model.pth")

    for epoch in range(cfg.TRAIN.EPOCH):
        if epoch < warmup_epochs:
            lr = lr_min + (lr_max - lr_min) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        train_results = run_one_epoch_cosface(net, train_loader, device, criterion, optimizer, mode="train")
        val_results   = run_one_epoch_cosface(net, val_loader, device, criterion, mode="eval")

        log_scalars(writer, logger, "Train", train_results, epoch)
        log_scalars(writer, logger, "Valid", val_results, epoch)

        # 保存最优模型
        if val_results["final_acc"] > best_acc + 1e-6:              
            best_acc = val_results["final_acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, ckpt_path)
            logger.info(f"✅ 保存最佳模型: {ckpt_path}")

        scheduler.step()

    writer.close()

    # 测试
    if cfg.USE_STAGE1:
        stage1_correct, idx_needed_net= stage1_test(test_idx, raw_data, best_params)
    else:
        stage1_correct = 0
        idx_needed_net = test_idx
    train_loader, val_loader, test_loader = build_dataloaders(
				cfg,
				train_idx=[],
				val_idx=[],
                test_idx=idx_needed_net,
				raw_data=raw_data,
				device=device
			)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])

    test_results  = run_one_epoch_cosface(net, test_loader, device, criterion, mode="test", save_dir=log_dir)
    stage2_correct = test_results["stage2_correct"]

    final_acc = (stage1_correct + stage2_correct) / len(test_idx)
    net_acc = test_results["final_acc"]

    logger.info(f"全系统测试结果: {final_acc}")
    logger.info(f"网络测试结果: {net_acc}")
    log_scalars(None, logger, "Test", test_results, "final")

if __name__ == "__main__":
    main()
