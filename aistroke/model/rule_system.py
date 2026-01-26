import numpy as np

def choose_angle_and_diff(raw_data, idx, min_angle_list, max_diff_list, purity_min):
    best_candidates = []
    best_yc2yc = -1

    for min_angle in min_angle_list:
        for max_diff in max_diff_list:
            yc2yc = 0  # 异常 -> 判异常
            zc2yc = 0  # 正常 -> 判异常

            for i in idx:
                sample = raw_data[i]
                gt = sample["total_label"]

                left = np.array(sample["left_arm_angles"])
                right = np.array(sample["right_arm_angles"])

                # 角度幅值过小 → 异常
                if np.mean(left) < min_angle or np.mean(right) < min_angle:
                    if gt == 0:
                        yc2yc += 1
                    else:
                        zc2yc += 1
                    continue

                # 左右差
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

    # 在 yc2yc 相同情况下，角度阈值越大越保守
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

        # 角度幅值过小
        if np.mean(left) < min_angle or np.mean(right) < min_angle:
            if gt == 0:
                yc2yc += 1
            else:
                zc2yc += 1
            continue

        # 左右差
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

def rule_system_train(idx, raw_data, cfg, logger):
    min_angle_list = range(0, 60)
    max_diff_list = range(0, 40)
    purity_min = cfg.TRAIN.RULE_SYSTEM.PURITY_MIN

    best_angle_params = choose_angle_and_diff(
        raw_data, idx, min_angle_list, max_diff_list, purity_min
    )

    yc2yc, zc2yc, _ = use_angle_threshold(
        raw_data, idx, best_angle_params
    )

    total_yc = sum(raw_data[i]["total_label"] == 0 for i in idx)

    recall_yc = yc2yc / (total_yc + 1e-6)
    purity = yc2yc / (yc2yc + zc2yc + 1e-6)

    logger.info("====== 规则系统 高置信异常检测（角度规则） ======")
    logger.info(f"min_angle_threshold = {best_angle_params[0]}")
    logger.info(f"max_diff_threshold  = {best_angle_params[1]}")
    logger.info(f"异常召回率 (Recall) = {recall_yc:.4f}")
    logger.info(f"异常纯度 (Precision)= {purity:.4f}")
    logger.info(f"规则系统剔除异常数量 = {yc2yc}")

    return best_angle_params

def rule_system_test(idx, raw_data, best_angle_params):
    yc2yc, _, idx_needed_net = use_angle_threshold(
        raw_data, idx, best_angle_params
    )

    return yc2yc, idx_needed_net
