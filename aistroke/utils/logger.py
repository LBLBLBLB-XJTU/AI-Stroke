import logging
import os

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

def log_scalars(writer, logger, prefix, data_dict, epoch):
    """
    将一个字典中的键值对依次写入 TensorBoard 和日志。

    参数:
    ----------
    writer : SummaryWriter or None
        TensorBoard 记录器。
    logger : logging.Logger
        控制台/文件输出。
    prefix : str
        前缀，如 "Train", "Valid"。
    data_dict : dict
        要记录的键值对，例如 {"Total_Loss": 0.123, "Status": "OK"}。
    epoch : int
        当前 epoch 编号。
    """

    # 写入 TensorBoard（仅记录可数值的）
    if writer is not None:
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):  # 仅限标量
                writer.add_scalar(f"{prefix}/{key}", value, epoch)

    # 写入日志：打印所有项（包括字符串）
    formatted_items = []
    for k, v in data_dict.items():
        if isinstance(v, (float, int)):
            formatted_items.append(f"{k}: {v:.4f}")
        else:
            formatted_items.append(f"{k}: {v}")

    msg = f"{prefix} {epoch} | " + " | ".join(formatted_items)
    logger.info(msg)
