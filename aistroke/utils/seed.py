import torch
import random
import numpy as np

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