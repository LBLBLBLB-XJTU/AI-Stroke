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

    # 单层交叉验证
    skf = StratifiedKFold(
        n_splits=cfg.CV_FOLDS,
        shuffle=True,
        random_state=cfg.SEED_VALUE
    )

    fold_acc = []
    fold_net_acc = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels)
    ):
        logger.info(f"===== Fold {fold} =====")
        fold_dir = os.path.join(log_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        ckpt_path = os.path.join(fold_dir, "final_model.pth")

        train_idx = list(train_idx)
        test_idx = list(test_idx)

        # =========================
        # Stage 1（规则/先验阶段）
        # =========================
        if cfg.USE_STAGE1:
            best_params = stage1_train(train_idx, raw_data, cfg)
            stage1_correct, idx_needed_net = stage1_test(
                test_idx, raw_data, best_params
            )
        else:
            stage1_correct = 0
            idx_needed_net = test_idx

        # =========================
        # DataLoader
        # =========================
        train_loader, _, test_loader = build_dataloaders(
            cfg,
            train_idx=train_idx,
            val_idx=[],
            test_idx=idx_needed_net,
            raw_data=raw_data,
            device=device
        )

        # =========================
        # Model / Optim / Loss
        # =========================
        net = MyNet(cfg).to(device)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.TRAIN.T_MAX,
            eta_min=1e-6
        )
        criterion = Losses(cfg)

        writer = SummaryWriter(log_dir=fold_dir)

        lr_max = cfg.TRAIN.LR
        lr_min = 1e-6

        # =========================
        # Train
        # =========================
        for epoch in range(cfg.TRAIN.EPOCH):
            if epoch < cfg.TRAIN.WARMUP_EPOCHS:
                lr = lr_min + (lr_max - lr_min) * (epoch + 1) / cfg.TRAIN.WARMUP_EPOCHS
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            train_results = run_one_epoch(
                net, train_loader, device, criterion,
                optimizer, mode="train"
            )

            log_scalars(writer, logger, "Train", train_results, epoch)
            scheduler.step()
        
        writer.close()

        torch.save(
            {
                "fold": fold,
                "epoch": cfg.TRAIN.EPOCH - 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
            },
            ckpt_path
        )
        logger.info(f"✅ Fold {fold} 最终模型已保存至: {ckpt_path}")

        # =========================
        # Test
        # =========================
        test_results = run_one_epoch(
            net,
            test_loader,
            device,
            criterion,
            mode="test",
            save_dir=fold_dir
        )

        stage2_correct = test_results["stage2_correct"]
        final_acc = (stage1_correct + stage2_correct) / len(test_idx)
        net_acc = test_results["final_acc"]

        logger.info(f"Fold {fold} 全系统准确率: {final_acc:.4f}")
        logger.info(f"Fold {fold} 网络准确率: {net_acc:.4f}")

        fold_acc.append(final_acc)
        fold_net_acc.append(net_acc)

    # =========================
    # Final Report
    # =========================
    logger.info(f"各折系统准确率: {fold_acc}")
    logger.info(f"各折网络准确率: {fold_net_acc}")
    logger.info(f"平均系统准确率: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    logger.info(f"平均网络准确率: {np.mean(fold_net_acc):.4f} ± {np.std(fold_net_acc):.4f}")

if __name__ == "__main__":
    main()
