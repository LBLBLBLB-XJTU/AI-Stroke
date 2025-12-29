import os
import os.path as osp
import sys
import time
import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random
import logging
from itertools import product
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

CLIPPED_DATA_PATH = "/home/liu_bang/AIStroke/raw_data_generate/raw_label_data_clipped.pkl"

def set_random_seed(seed: int = 42):
    """
    固定所有随机种子，确保结果可复现。
    包括 Python、NumPy、PyTorch（CPU & GPU）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 保证 cudnn 结果确定（但可能略慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir):
    """
    创建 logger，控制台 + 文件输出
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清空之前的 handler 避免重复日志
    logger.log_dir = log_dir

    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    # 控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 文件
    file_handler = logging.FileHandler(os.path.join(log_dir, "log.log"), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger

from numba import njit
import numpy as np

@njit
def check_interval_jit(left, right, s, e, a_t, d_t):
    """
    判断 [s, e) 区间是否满足规则
    正常返回 True
    """
    for i in range(s, e):
        if left[i] < a_t or right[i] < a_t:
            return False
        if abs(left[i] - right[i]) > d_t:
            return False
    return True

def build_prefix(arr):
    arr = np.asarray(arr, dtype=np.float64)
    ps = np.zeros(len(arr) + 1)
    ps2 = np.zeros(len(arr) + 1)
    ps[1:] = np.cumsum(arr)
    ps2[1:] = np.cumsum(arr ** 2)
    return ps, ps2

def calc_std(ps, ps2, l, r):
    n = r - l
    mean = (ps[r] - ps[l]) / n
    var = (ps2[r] - ps2[l]) / n - mean ** 2
    return np.sqrt(max(var, 0.0))

def find_longest_stable_interval(left, right, metric, s_t, w_t):
    T = min(len(left), len(right))
    lps, lps2 = build_prefix(left)
    rps, rps2 = build_prefix(right)

    best_len = 0
    best_range = None

    start = 0
    end = 0

    while start + w_t <= T:
        if end < start + w_t:
            end = start + w_t

        while end <= T:
            if metric == "sigma":
                lv = calc_std(lps, lps2, start, end)
                rv = calc_std(rps, rps2, start, end)
            elif metric == "sigma^2":
                lv = calc_std(lps, lps2, start, end) ** 2
                rv = calc_std(rps, rps2, start, end) ** 2
            else:  # sigma^4
                lv = calc_std(lps, lps2, start, end) ** 4
                rv = calc_std(rps, rps2, start, end) ** 4

            if lv <= s_t and rv <= s_t:
                if end - start > best_len:
                    best_len = end - start
                    best_range = (start, end)
                end += 1
            else:
                break

        start += 1

    return best_range

def evaluate_one_setting(
    data,
    intervals,
    a_t,
    d_t
):
    pos_c = pos_t = neg_c = neg_t = 0

    for sample, interval in zip(data, intervals):
        # GT：正常=1，患病=0
        gt = 1
        if sample["left_label"] == 1 or sample["right_label"] == 1:
            gt = 0

        pred = 0
        if interval is not None:
            s, e = interval
            left = np.asarray(sample["left_arm_angles"], dtype=np.float64)
            right = np.asarray(sample["right_arm_angles"], dtype=np.float64)

            if check_interval_jit(left, right, s, e, a_t, d_t):
                pred = 1

        if gt == 1:
            pos_t += 1
            if pred == gt:
                pos_c += 1
        else:
            neg_t += 1
            if pred == gt:
                neg_c += 1

    acc_pos = pos_c / pos_t if pos_t else 0.0
    acc_neg = neg_c / neg_t if neg_t else 0.0
    return acc_pos, acc_neg

from joblib import Parallel, delayed
from itertools import product

def get_best_rule(data, n_jobs=8):

    stability_metric_list = ["sigma", "sigma^2", "sigma^4"]
    stability_threshold_range = np.linspace(1.0, 8.0, 8)
    window_len_range = [10, 15, 20, 25, 30]

    arm_angle_threshold_range = np.arange(20, 91, 5)
    diff_threshold_range = np.arange(5, 31, 2)

    # ========= 1. 预计算稳定区间 =========
    stable_cache = {}

    for metric, s_t, w_t in product(
        stability_metric_list,
        stability_threshold_range,
        window_len_range
    ):
        intervals = [
            find_longest_stable_interval(
                sample["left_arm_angles"],
                sample["right_arm_angles"],
                metric, s_t, w_t
            )
            for sample in data
        ]
        stable_cache[(metric, s_t, w_t)] = intervals

    # ========= 2. 并行搜索角度规则 =========
    best_score = -1
    best_rule = None

    for metric, s_t, w_t in product(
        stability_metric_list,
        stability_threshold_range,
        window_len_range
    ):
        intervals = stable_cache[(metric, s_t, w_t)]

        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_one_setting)(
                data, intervals, a_t, d_t
            )
            for a_t, d_t in product(
                arm_angle_threshold_range,
                diff_threshold_range
            )
        )

        for (a_t, d_t), (acc_pos, acc_neg) in zip(
            product(arm_angle_threshold_range, diff_threshold_range),
            results
        ):
            score = 0.5 * (acc_pos + acc_neg)
            print(f"{score}")
            if score > best_score:
                best_score = score
                best_rule = {
                    "metric": metric,
                    "stability_threshold": s_t,
                    "window_len": w_t,
                    "arm_angle_threshold": a_t,
                    "diff_threshold": d_t,
                    "acc_pos": acc_pos,
                    "acc_neg": acc_neg,
                    "score": score
                }

    return best_rule

def get_test_acc(rule, data):
    """
    使用最优规则在测试集上计算准确率
    正常 = 1，患病 = 0
    """

    metric = rule["metric"]
    s_t = rule["stability_threshold"]
    w_t = rule["window_len"]
    a_t = rule["arm_angle_threshold"]
    d_t = rule["diff_threshold"]

    correct = 0
    total = len(data)

    for sample in data:
        left_angles = np.asarray(sample["left_arm_angles"], dtype=np.float32)
        right_angles = np.asarray(sample["right_arm_angles"], dtype=np.float32)

        # ===== 1. 找稳定区间 =====
        interval = find_longest_stable_interval(
            left_angles,
            right_angles,
            metric,
            s_t,
            w_t
        )

        # 没有稳定区间 → 判患病
        if interval is None:
            pred = 0
        else:
            start, end = interval
            l_seg = left_angles[start:end]
            r_seg = right_angles[start:end]

            # ===== 2. 角度 & 差值规则 =====
            angle_ok = np.all((l_seg >= a_t) & (r_seg >= a_t))
            diff_ok = np.all(np.abs(l_seg - r_seg) <= d_t)

            pred = 1 if (angle_ok and diff_ok) else 0

        # ===== 3. GT（左右只要一侧患病即患病）=====
        gt = 1 if (sample["left_label"] == 1 and sample["right_label"] == 1) else 0

        if pred == gt:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return acc

def main():
    set_random_seed(1)

    # 日志
    logs_root = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_root, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    log_dir = os.path.join(logs_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # 加载数据
    raw_data = joblib.load(CLIPPED_DATA_PATH)
    labels = [d["total_label"] for d in raw_data]

    # 简单划分
    test_acc_list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):

        train_idx = list(train_idx)
        test_idx = list(test_idx)
        train_data = [raw_data[i] for i in train_idx]
        test_data = [raw_data[i] for i in test_idx]

        best_rule = get_best_rule(train_data)
        test_acc = get_test_acc(best_rule, test_data)
        logger.info(f"fold{fold}的正确率为{test_acc}")
        test_acc_list.append(test_acc)

    logger.info(test_acc_list)
    logger.info(f"平均正确率{sum(test_acc_list) / len(test_acc_list)}")

if __name__ == "__main__":
    main()