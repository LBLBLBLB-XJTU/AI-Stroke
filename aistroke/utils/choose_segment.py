
def choose_segment(raw_data, cfg):
    segment_option = cfg.DATA.SEGMENT
    new_raw_data = []
    
    if segment_option == 0:
        for sample in raw_data:
            new_sample = {
                "id": sample["id"],
                "joints": sample["joints_0"],
                "left_arm_angles": sample["left_arm_angles_0"],
                "right_arm_angles": sample["right_arm_angles_0"],
                "total_label": sample["total_label"],
                "left_label": sample["left_label"],
                "right_label": sample["right_label"],
            }
            new_raw_data.append(new_sample)
    elif segment_option == 1:
        for sample in raw_data:
            new_sample = {
                "id": sample["id"],
                "joints": sample["joints_1"],
                "left_arm_angles": sample["left_arm_angles_1"],
                "right_arm_angles": sample["right_arm_angles_1"],
                "total_label": sample["total_label"],
                "left_label": sample["left_label"],
                "right_label": sample["right_label"],
            }
            new_raw_data.append(new_sample)
    elif segment_option == 2:
        for sample in raw_data:
            new_sample = {
                "id": sample["id"],
                "joints": sample["joints_2"],
                "left_arm_angles": sample["left_arm_angles_2"],
                "right_arm_angles": sample["right_arm_angles_2"],
                "total_label": sample["total_label"],
                "left_label": sample["left_label"],
                "right_label": sample["right_label"],
            }
            new_raw_data.append(new_sample)
    else:
        raise ValueError(f"Unknown segment option: {segment_option}")
    return new_raw_data