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
cfg.SPLIT_RATIO = [80, 10, 10]
cfg.USE_STAGE1 = False

# 路径配置
cfg.PATH = CN()
cfg.PATH.RAW_LABEL_DATA_PATH = "raw_data_generate/raw_label_data_clipped_byaudio.pkl"

# 训练参数
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR = 1e-5
cfg.TRAIN.WEIGHT_DECAY = cfg.TRAIN.LR * 0.1
cfg.TRAIN.T_MAX = 1000

# 数据参数
cfg.DATA = CN() 
cfg.DATA.AUGMENT = CN()
cfg.DATA.AUGMENT.AUGMENT_PROB = 0
cfg.DATA.AUGMENT.NOISE_STD = 0.01
cfg.DATA.AUGMENT.TRANSLATE_RANGE = 0.05
cfg.DATA.AUGMENT.SCALE_RANGE = (0.9, 1.1)
cfg.DATA.AUGMENT.ROTATE_RANGE_DENOMINATOR = 18

# loss参数
cfg.LOSS = CN()
cfg.LOSS.LABEL_SMOOTHING = 0.3
cfg.LOSS.CE_LAMBDA = 1
cfg.LOSS.L2_LAMBDA = 2e-4
cfg.LOSS.FEAT_LAMBDA = 5e-2
cfg.LOSS.CENTER_LAMBDA = 1e-2
cfg.LOSS.TRIPLET_LAMBDA = 1e-2
cfg.LOSS.TRIPLET_MARGIN = 0.3

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
    48,  
    1, 
    1,
    1
)
cfg.MODEL.MODALITIES_USED_IDX = [0, 1, 2, 3] # 确定
# 模型参数
cfg.MODEL.EMBED_DIM = 256
cfg.MODEL.TRANSFORMER_LAYERS = 3
cfg.MODEL.DROP = 0.3
cfg.MODEL.NUM_HEADS = 8
# 骨架配置
cfg.MODEL.SKELETON_NAME = "joints"
cfg.MODEL.NUM_JOINTS = 16
cfg.MODEL.SKELETON_CHANNELS = 3
cfg.MODEL.SKELETON_NUM_HEADS = max(1, cfg.MODEL.NUM_HEADS // 2)
cfg.MODEL.SKELETON_CONV_CHANNELS = [64, 128]
cfg.MODEL.SKELETON_TIME_POOL_TO = 32
# 其他模态配置
cfg.MODEL.OTHER_CONV_CHANNELS = [64, 128]
cfg.MODEL.OTHER_TIME_POOL_TO = 32
# 分类器配置
cfg.MODEL.FACE_S = 30.0
cfg.MODEL.FACE_M = 0.35

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