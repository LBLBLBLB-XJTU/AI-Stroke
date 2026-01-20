import os
import sys
import time
import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from collections import Counter

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

    logs_root = os.path.join(cfg.PROJECT_ROOT, "logs")
    os.makedirs(logs_root, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    log_dir = os.path.join(logs_root, f"CV_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)
    logger.info(f"配置：{cfg}")

    # 加载数据
    raw_path = os.path.join(os.path.dirname(cfg.PROJECT_ROOT), cfg.PATH.DATA_PATH)
    raw_data = joblib.load(raw_path)
    raw_data = choose_segment(raw_data, cfg)
    labels = np.array([d["total_label"] for d in raw_data])

    logger.info(f"总样本数: {len(labels)}")
    logger.info(f"类别分布: {Counter(labels)}")

    # ========= K 折 =========
    skf = StratifiedKFold(
        n_splits=cfg.CV_FOLDS,
        shuffle=True,
        random_state=cfg.SEED_VALUE
    )

    all_fold_results = []

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_dir = os.path.join(log_dir, f"fold_{fold}")
        ckpt_dir = os.path.join(fold_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)

        fold_logger = setup_logger(fold_dir)
        fold_logger.info(f"========== Fold {fold} ==========")

        # ========= Stage1 =========
        best_stage1_params = stage1_train(trainval_idx, raw_data, cfg, fold_logger)
        fold_logger.info(f"stage1参数选择：{best_stage1_params}")

        train_loader, val_loader, _ = build_dataloaders(
            cfg,
            train_idx=trainval_idx,
            val_idx=trainval_idx,
            test_idx=[],
            raw_data=raw_data,
            device=device
        )

        net = MyNet(cfg).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.T_MAX, eta_min=1e-6)
        criterion = Losses(cfg)
        lr_max = cfg.TRAIN.LR
        lr_min = 1e-6

        saved_ckpts = []

        for epoch in range(cfg.TRAIN.EPOCH):
            if epoch < cfg.TRAIN.WARMUP_EPOCHS:
                lr = lr_min + (lr_max - lr_min) * (epoch + 1) / cfg.TRAIN.WARMUP_EPOCHS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            train_results = run_one_epoch(
                net, train_loader, device, criterion, optimizer, mode="train"
            )

            scheduler.step()

            # ---- ckpt 保存策略 ----
            is_save_epoch = (
                (epoch + 1) % 20 == 0 or
                (epoch + 1) == cfg.TRAIN.EPOCH
            )

            if is_save_epoch:
                ckpt_path = os.path.join(
                    ckpt_dir, f"epoch_{epoch + 1}.pth"
                )
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, ckpt_path)

                saved_ckpts.append(ckpt_path)
                fold_logger.info(f"💾 保存 ckpt: {ckpt_path}")

        # ========= 测试（每个 ckpt） =========
        fold_results = []

        if cfg.USE_STAGE1:
            stage1_correct, idx_needed_net = stage1_test(test_idx, raw_data, best_stage1_params)
        else:
            stage1_correct = 0
            idx_needed_net = test_idx

        _, _, test_loader = build_dataloaders(
            cfg,
            train_idx=[],
            val_idx=[],
            test_idx=idx_needed_net,
            raw_data=raw_data,
            device=device
        )

        for ckpt_path in saved_ckpts:
            checkpoint = torch.load(ckpt_path, map_location=device)
            net.load_state_dict(checkpoint["model_state_dict"])

            test_results = run_one_epoch(net, test_loader, device, criterion, mode="test", save_dir=fold_dir)

            stage2_correct = test_results["stage2_correct"]
            final_acc = (stage1_correct + stage2_correct) / len(test_idx)

            record = {
                "fold": fold,
                "epoch": checkpoint["epoch"],
                "final_acc": final_acc,
                "net_acc": test_results["final_acc"]
            }
            fold_results.append(record)

            fold_logger.info(
                f"[Fold {fold}] Epoch {checkpoint['epoch']} | "
                f"Final Acc: {final_acc:.4f}, Net Acc: {test_results['final_acc']:.4f}"
            )

        all_fold_results.extend(fold_results)

    logger.info("========== Cross Validation Summary ==========")
    for r in all_fold_results:
        logger.info(r)


if __name__ == "__main__":
    main()
