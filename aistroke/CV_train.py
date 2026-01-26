import os
import time
import joblib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from .configs.config import parse_args
from .utils.seed import set_random_seed
from .utils.logger import setup_logger, log_scalars
from .utils.choose_segment import choose_segment
from .model.rule_system import rule_system_train, rule_system_test
from .dataset.dataset import build_dataloaders
from .model.MyNet import MyNet
from .model.loss import Losses
from .utils.engine import run_one_epoch

def CV_train(args, PROJECT_ROOT):
    cfg = parse_args(args, PROJECT_ROOT)
    set_random_seed(cfg.SEED_VALUE)
    device = torch.device(cfg.DEVICE)

    # 日志
    logs_root = os.path.join(cfg.PROJECT_ROOT, "logs")
    os.makedirs(logs_root, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    cv_log_root = os.path.join(logs_root, f"CV_{timestamp}")
    os.makedirs(cv_log_root, exist_ok=True)
    logger = setup_logger(cv_log_root)
    logger.info(f"配置：{cfg}")

    # 加载数据
    raw_path = os.path.join(cfg.PROJECT_ROOT, cfg.PATH.DATA_PATH)
    raw_data = joblib.load(raw_path)
    if cfg.DATA.SEGMENT.USE_SEGMENT:
        raw_data = choose_segment(raw_data, cfg)
    labels = np.array([d["total_label"] for d in raw_data])
    counter = Counter(labels)
    logger.info("类别分布：")
    for k, v in counter.items():
        logger.info(f"  label={k}: {v} 个样本")

    # K-Fold
    n_folds = cfg.CV_FOLDS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED_VALUE)

    use_rule = cfg.TRAIN.RULE_SYSTEM.USE_RULE_SYSTEM
    if use_rule:
        fold_results_bestckpt_allsystem_acc = []
        fold_results_bestckpt_net_acc = []
        fold_results_lastckpt_allsystem_acc = []
        fold_results_lastckpt_net_acc = []
    else:
        fold_results_bestckpt_acc = []
        fold_results_lastckpt_acc = []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info("=" * 60)
        logger.info(f"🚀 Fold [{fold + 1}/{n_folds}]")

        fold_dir = os.path.join(cv_log_root, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_dir)

        # 再切 val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=cfg.SPLIT_RATIO[1] / (cfg.SPLIT_RATIO[0] + cfg.SPLIT_RATIO[1]),
            stratify=labels[train_val_idx],
            random_state=cfg.SEED_VALUE
        )

        logger.info(f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # ---------- rule system ----------
        if use_rule:
            best_params = rule_system_train(train_idx, raw_data, cfg, logger)
            logger.info(f"规则系统参数选择：{best_params}")

        # ---------- dataloader ----------
        train_loader, val_loader, _ = build_dataloaders(
            cfg,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=[],
            raw_data=raw_data,
            device=device
        )

        # ---------- model ----------
        net = MyNet(cfg).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.T_MAX, eta_min=1e-6)
        criterion = Losses(cfg)
        lr_max = cfg.TRAIN.LR
        lr_min = 1e-6

        best_acc = 0.0
        ckpt_path = os.path.join(fold_dir, "best_model.pth")
        last_ckpt_path = os.path.join(fold_dir, "last_model.pth")

        # ================= 训练 =================
        for epoch in range(cfg.TRAIN.EPOCH):
            if epoch < cfg.TRAIN.WARMUP_EPOCHS:
                lr = lr_min + (lr_max - lr_min) * (epoch + 1) / cfg.TRAIN.WARMUP_EPOCHS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            train_results = run_one_epoch(net, train_loader, device, criterion, optimizer, mode="train")
            val_results = run_one_epoch(net, val_loader, device, criterion, mode="eval")

            log_scalars(writer, logger, "Train", train_results, epoch)
            log_scalars(writer, logger, "Valid", val_results, epoch)

            if val_results["final_acc"] > best_acc:
                best_acc = val_results["final_acc"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                }, ckpt_path)
                logger.info(f"✅ 保存最佳模型: {ckpt_path}")

            scheduler.step()

        # 保存最后模型
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }, last_ckpt_path)

        writer.close()

        # 测试
        if use_rule:
            rule_system_correct, idx_needed_net = rule_system_test(test_idx, raw_data, best_params)
        else:
            rule_system_correct = 0
            idx_needed_net = test_idx
        _, _, test_loader = build_dataloaders(
            cfg,
            train_idx=[],
            val_idx=[],
            test_idx=idx_needed_net,
            raw_data=raw_data,
            device=device
        )

        checkpoints = [("Test", ckpt_path), ("Test_Last_ckpt", last_ckpt_path)]

        for tag, path in checkpoints:
            checkpoint = torch.load(path, map_location=device)
            net.load_state_dict(checkpoint["model_state_dict"])
            test_results = run_one_epoch(net, test_loader, device, criterion, mode="test", save_dir=fold_dir)

            net_correct = test_results["net_correct"]
            net_acc = test_results["final_acc"]

            if use_rule:
                final_acc = (rule_system_correct + net_correct) / len(test_idx)
                logger.info(f"{tag} 全系统测试结果: {final_acc:.4f}")
                logger.info(f"{tag} 网络测试结果: {net_acc:.4f}")
                if tag == "Test":
                    fold_results_bestckpt_allsystem_acc.append(final_acc)
                    fold_results_bestckpt_net_acc.append(net_acc)
                else:
                    fold_results_lastckpt_allsystem_acc.append(final_acc)
                    fold_results_lastckpt_net_acc.append(net_acc)
            else:
                logger.info(f"{tag} 测试结果: {net_acc:.4f}")
                if tag == "Test":
                    fold_results_bestckpt_acc.append(net_acc)
                else:
                    fold_results_lastckpt_acc.append(net_acc)

            log_scalars(None, logger, tag, test_results, "final")

    # ================= 汇总 =================
    if use_rule:
        fold_results_bestckpt_allsystem_acc = np.array(fold_results_bestckpt_allsystem_acc)
        fold_results_bestckpt_net_acc = np.array(fold_results_bestckpt_net_acc)
        fold_results_lastckpt_allsystem_acc = np.array(fold_results_lastckpt_allsystem_acc)
        fold_results_lastckpt_net_acc = np.array(fold_results_lastckpt_net_acc)
        logger.info(f"🎯 验证最佳ckpt CV Result on 全系统: {fold_results_bestckpt_allsystem_acc.mean():.4f} ± {fold_results_bestckpt_allsystem_acc.std():.4f}")
        logger.info(f"🎯 验证最佳ckpt CV Result on 仅网络: {fold_results_bestckpt_net_acc.mean():.4f} ± {fold_results_bestckpt_net_acc.std():.4f}")
        logger.info(f"🎯 验证最后ckpt CV Result on 全系统: {fold_results_lastckpt_allsystem_acc.mean():.4f} ± {fold_results_lastckpt_allsystem_acc.std():.4f}")
        logger.info(f"🎯 验证最后ckpt CV Result on 仅网络: {fold_results_lastckpt_net_acc.mean():.4f} ± {fold_results_lastckpt_net_acc.std():.4f}")
    else:
        fold_results_bestckpt_acc = np.array(fold_results_bestckpt_acc)
        fold_results_lastckpt_acc = np.array(fold_results_lastckpt_acc)
        logger.info(f"🎯 验证最佳ckpt CV Result: {fold_results_bestckpt_acc.mean():.4f} ± {fold_results_bestckpt_acc.std():.4f}")
        logger.info(f"🎯 验证最后ckpt CV Result: {fold_results_lastckpt_acc.mean():.4f} ± {fold_results_lastckpt_acc.std():.4f}")