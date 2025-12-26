import torch
import random

def joints_augment(joints, cfg):
    """
    对 SMPL 骨架序列进行数据增强（GPU版 + 概率控制）
    joints: torch.Tensor, shape (T, 24, 3) 或 (B, T, 24, 3)
    """
    device = joints.device
    joints_aug = joints.clone()
    prob = cfg.DATA.AUGMENT.AUGMENT_PROB
    noise_std = cfg.DATA.AUGMENT.NOISE_STD
    translate_range = cfg.DATA.AUGMENT.TRANSLATE_RANGE
    scale_range = cfg.DATA.AUGMENT.SCALE_RANGE
    rotate_range = torch.pi / cfg.DATA.AUGMENT.ROTATE_RANGE_DENOMINATOR
    
    # 1. 加噪声
    if random.random() < prob:
        joints_aug += torch.randn_like(joints_aug) * noise_std
    
    # 2. 随机平移
    if random.random() < prob:
        translation = (torch.rand(1,1,3, device=device) * 2 - 1) * translate_range
        if joints_aug.dim() == 4:  # B, T, 24, 3
            translation = translation.unsqueeze(0)  # (1,1,1,3)
        joints_aug += translation
    
    # 3. 随机旋转（绕Y轴）
    if random.random() < prob:
        angle = (torch.rand(1, device=device) * 2 - 1) * rotate_range
        cosval, sinval = torch.cos(angle), torch.sin(angle)
        R = torch.tensor([[cosval, 0, sinval],
                          [0,      1, 0],
                          [-sinval,0, cosval]], device=device)
        joints_aug = torch.matmul(joints_aug, R.T)
    
    # 4. 随机缩放
    if random.random() < prob:
        scale = torch.empty(1, device=device).uniform_(scale_range[0], scale_range[1])
        joints_aug *= scale
    
    return joints_aug
