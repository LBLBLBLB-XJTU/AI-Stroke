import numpy as np

# =====================================================
# 1. 时间长度阈值选择
# =====================================================
def choose_time_threshold(raw_data, idx, time_threshold_list, purity_min):
    best_time_threshold = None
    best_yc2yc = -1

    for time_threshold in time_threshold_list:
        yc2yc = 0  # 异常 -> 判异常
        zc2yc = 0  # 正常 -> 判异常

        for i in idx:
            sample = raw_data[i]
            L = len(sample["left_arm_angles"])

            if L < time_threshold * 30:  # 判异常
                if sample["total_label"] == 0:
                    yc2yc += 1
                else:
                    zc2yc += 1

        purity = yc2yc / (yc2yc + zc2yc + 1e-6)
        if purity > purity_min and yc2yc > best_yc2yc:
            best_yc2yc = yc2yc
            best_time_threshold = time_threshold

    return best_time_threshold


def use_time_threshold(raw_data, idx, best_time_threshold):
    yc2yc = 0
    zc2yc = 0
    idx_needed_angle = []

    for i in idx:
        sample = raw_data[i]
        L = len(sample["left_arm_angles"])

        if L < best_time_threshold * 30:  # 判异常
            if sample["total_label"] == 0:
                yc2yc += 1
            else:
                zc2yc += 1
        else:
            idx_needed_angle.append(i)

    return yc2yc, zc2yc, idx_needed_angle


# =====================================================
# 2. 角度 & 左右差阈值选择
# =====================================================
def choose_angle_and_diff(raw_data, idx, min_angle_list, max_diff_list, purity_min):
    best_candidates = []
    best_yc2yc = -1

    for min_angle in min_angle_list:
        for max_diff in max_diff_list:
            yc2yc = 0
            zc2yc = 0

            for i in idx:
                sample = raw_data[i]
                gt = sample["total_label"]

                left = np.array(sample["left_arm_angles"])
                right = np.array(sample["right_arm_angles"])

                if np.mean(left) < min_angle or np.mean(right) < min_angle:
                    if gt == 0:
                        yc2yc += 1
                    else:
                        zc2yc += 1
                    continue

                L = min(len(left), len(right))
                diff_mean = np.mean(np.abs(left[:L] - right[:L]))

                if diff_mean > max_diff:
                    if gt == 0:
                        yc2yc += 1
                    else:
                        zc2yc += 1

            purity = yc2yc / (yc2yc + zc2yc + 1e-6)
            if purity > purity_min:
                if yc2yc > best_yc2yc:
                    best_yc2yc = yc2yc
                    best_candidates = [(min_angle, max_diff)]
                elif yc2yc == best_yc2yc:
                    best_candidates.append((min_angle, max_diff))

    # 角度越大越保守
    return max(best_candidates, key=lambda x: x[0])


def use_angle_threshold(raw_data, idx, best_angle_params):
    min_angle, max_diff = best_angle_params

    yc2yc = 0
    zc2yc = 0
    idx_needed_net = []

    for i in idx:
        sample = raw_data[i]
        gt = sample["total_label"]

        left = np.array(sample["left_arm_angles"])
        right = np.array(sample["right_arm_angles"])

        if np.mean(left) < min_angle or np.mean(right) < min_angle:
            if gt == 0:
                yc2yc += 1
            else:
                zc2yc += 1
            continue

        L = min(len(left), len(right))
        diff_mean = np.mean(np.abs(left[:L] - right[:L]))

        if diff_mean > max_diff:
            if gt == 0:
                yc2yc += 1
            else:
                zc2yc += 1
            continue

        idx_needed_net.append(i)

    return yc2yc, zc2yc, idx_needed_net


# =====================================================
# 3. Stage1 训练（选阈值）
# =====================================================
def stage1_train(idx, raw_data, cfg, logger):
    time_threshold_list = np.arange(2, 5, 0.1)
    min_angle_list = range(0, 60)
    max_diff_list = range(0, 40)

    purity_min = cfg.STAGE1.PURITY_MIN

    # --- 时间阈值 ---
    best_time_threshold = choose_time_threshold(
        raw_data, idx, time_threshold_list, purity_min
    )
    t_yc2yc, t_zc2yc, idx_needed_angle = use_time_threshold(
        raw_data, idx, best_time_threshold
    )

    # --- 角度阈值 ---
    best_angle_params = choose_angle_and_diff(
        raw_data, idx_needed_angle, min_angle_list, max_diff_list, purity_min
    )
    a_yc2yc, a_zc2yc, _ = use_angle_threshold(
        raw_data, idx_needed_angle, best_angle_params
    )

    total_yc = sum(raw_data[i]["total_label"] == 0 for i in idx)

    recall_yc = (t_yc2yc + a_yc2yc) / (total_yc + 1e-6)
    purity = (t_yc2yc + a_yc2yc) / (
        t_yc2yc + a_yc2yc + t_zc2yc + a_zc2yc + 1e-6
    )

    logger.info("====== Stage1 高置信异常检测 ======")
    logger.info(f"min_time_threshold  = {best_time_threshold}")
    logger.info(f"min_angle_threshold = {best_angle_params[0]}")
    logger.info(f"max_diff_threshold  = {best_angle_params[1]}")
    logger.info(f"异常召回率 (Recall) = {recall_yc:.4f}")
    logger.info(f"异常纯度 (Precision)= {purity:.4f}")
    logger.info(f"Stage1 剔除异常数量 = {t_yc2yc + a_yc2yc}")

    return best_time_threshold, best_angle_params


# =====================================================
# 4. Stage1 测试 / 推理
# =====================================================
def stage1_test(idx, raw_data, best_params):
    best_time_threshold, best_angle_params = best_params

    t_yc2yc, _, idx_needed_angle = use_time_threshold(
        raw_data, idx, best_time_threshold
    )
    a_yc2yc, _, idx_needed_net = use_angle_threshold(
        raw_data, idx_needed_angle, best_angle_params
    )

    return t_yc2yc + a_yc2yc, idx_needed_net
