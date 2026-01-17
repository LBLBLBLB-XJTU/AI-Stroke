from yacs.config import CfgNode as CN

# Configuration variable
cfg = CN()

# 通用配置
cfg.TITLE = 'task2'
cfg.DEVICE = 'cuda'
cfg.SEED_VALUE = 0
cfg.PROJECT_ROOT = ""
cfg.OUTER_FOLDS = 10
cfg.INNER_FOLDS = 9
cfg.CV_FOLDS = 5
cfg.SPLIT_RATIO = [80, 10, 10]
cfg.USE_STAGE1 = True

cfg.STAGE1 = CN()
cfg.STAGE1.PURITY_MIN = 0.9 # 正常样本误杀容忍度

# 路径配置
cfg.PATH = CN()
cfg.PATH.DATA_PATH = "data_generate/3data_with_angles.pkl"
cfg.PATH.CLASS1_PATH = "data_generate/sample_txts/class_1.txt"

# 训练参数
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.EPOCH = 75
cfg.TRAIN.LR = 3e-4
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.T_MAX = 100
cfg.TRAIN.WARMUP_EPOCHS = 10

# 数据参数 目前没用到，先不管
cfg.DATA = CN()
cfg.DATA.SEGMENT = 2

# loss参数
cfg.LOSS = CN()
cfg.LOSS.LABEL_SMOOTHING = 0.05
cfg.LOSS.CE_LAMBDA = 1.0
cfg.LOSS.L2_LAMBDA = 1e-4
cfg.LOSS.CENTER_LAMBDA = 1e-4

# 模型参数
cfg.MODEL = CN()
# 模态选择
cfg.MODEL.MODALITIES_NAMES = (
    "joints", 
    "left_arm_angle", 
    "right_arm_angle",
    "diff"
)
cfg.MODEL.MODALITIES_DIMS = (
    33,  
    1, 
    1,
    1
)
cfg.MODEL.MODALITIES_USED_IDX = [0, 1, 2, 3]  # 确定
# 模型通用参数
cfg.MODEL.EMBED_DIM = 256
cfg.MODEL.DROP = 0.2
cfg.MODEL.NUM_HEADS = 4
cfg.MODEL.TOKEN_CONV_CHANNELS = [64, 128]
cfg.MODEL.TEMPORAL_STRIDE = [2, 2]
cfg.MODEL.MAX_T = 512
# 骨架配置
cfg.MODEL.SKELETON_NAME = "joints"
cfg.MODEL.SKELETON_CHANNELS = 3
cfg.MODEL.SKELETON_NUM_JOINTS = 11
# 时序配置
cfg.MODEL.NUM_TIME_HEADS = 4

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def get_cfg():
    """
    Define configuration.
    """
    cfg = get_cfg_defaults()
    return cfg.clone()

def parse_args():
    cfg = get_cfg()
    return cfg