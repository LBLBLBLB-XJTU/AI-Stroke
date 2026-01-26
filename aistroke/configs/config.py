import os
import os.path as osp
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

# 数据配置
cfg.DATA = CN()
cfg.DATA.DATASET_NAME = ""
cfg.DATA.SEGMENT = CN()
cfg.DATA.SEGMENT.USE_SEGMENT = True
cfg.DATA.SEGMENT.SEGMENT_OPTION = 0

# 路径配置
cfg.PATH = CN()
cfg.PATH.DATA_PATH = ""

# 训练参数
cfg.TRAIN = CN()
cfg.TRAIN.USE_STAGE1 = True
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.EPOCH = 100
cfg.TRAIN.LR = 3e-4
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.T_MAX = 100
cfg.TRAIN.WARMUP_EPOCHS = 10
cfg.TRAIN.RULE_SYSTEM = CN()
cfg.TRAIN.RULE_SYSTEM.USE_RULE_SYSTEM = True
cfg.TRAIN.RULE_SYSTEM.PURITY_MIN = 0.9

# 模型参数
cfg.MODEL = CN()
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
cfg.MODEL.SKELETON = CN()
cfg.MODEL.SKELETON.SKELETON_NAME = "joints"
cfg.MODEL.SKELETON.SKELETON_CHANNELS = 3
cfg.MODEL.SKELETON.SKELETON_NUM_JOINTS = 11
cfg.MODEL.SKELETON.ROOTIDX = 0
cfg.MODEL.SKELETON.HEAD_IDX = 4
cfg.MODEL.SKELETON.LEFT_SHOULDER_IDX = 5
cfg.MODEL.SKELETON.RIGHT_SHOULDER_IDX = 8
cfg.MODEL.SKELETON.LEFT_WRIST_IDX = 7
cfg.MODEL.SKELETON.RIGHT_WRIST_IDX = 10
cfg.MODEL.SKELETON.NECK_IDX = 3

# 时序配置
cfg.MODEL.NUM_TIME_HEADS = 4

# loss参数
cfg.LOSS = CN()
cfg.LOSS.LABEL_SMOOTHING = 0.05
cfg.LOSS.CE_LAMBDA = 1.0
cfg.LOSS.L2_LAMBDA = 1e-4
cfg.LOSS.CENTER_LAMBDA = 1e-4

def get_cfg_defaults():
    return cfg.clone()

def get_cfg(args, PROJECT_ROOT):
    cfg = get_cfg_defaults()
    cfg.PROJECT_ROOT = PROJECT_ROOT
    cfg_file_path = osp.join(PROJECT_ROOT, args.cfg)
    if os.path.exists(cfg_file_path):
        cfg.merge_from_file(cfg_file_path)

    return cfg.clone()

def parse_args(args, PROJECT_ROOT):
    return get_cfg(args, PROJECT_ROOT)