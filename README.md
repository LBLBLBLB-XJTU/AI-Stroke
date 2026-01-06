# AIStroke 项目说明

## 项目简介
AIStroke 是一个用于中风（脑卒中）相关动作识别与分析的深度学习项目，包含数据采集、特征生成、模型训练与推理等完整流程。项目支持多模态数据（如关节点、角度、音频等），并集成了多种深度学习模型结构和损失函数，适用于医学动作分析、康复评估等场景。

## 目录结构
```
AIStroke/
├── Test-And-Record-Demand.txt         # 样本测试与记录需求说明
├── aistroke/
│   ├── CT_CV_main.py                  # 主训练（支持交叉验证）
│   ├── main.py                        # 主训练
│   ├── stage1.py                      # 阶段一训练
│   ├── config/                        # 配置模块
│   ├── data/                          # 数据处理与特征生成
│   ├── logs/                          # 日志与模型权重保存
│   ├── model/                         # 模型结构与损失函数
│   └── utils/                         # 工具函数
├── raw_data_generate/                 # 原始数据处理与生成
│   ├── 1data_collect_joints.py        # 关节点数据采集
│   ├── 2data_collect_label.py         # 标签采集
│   ├── 3data_collect.py               # 综合数据采集
│   ├── 4get_angles.py                 # 角度特征提取
│   ├── 5.1clip_by_angles.py           # 按角度裁剪
│   ├── 5.2clip_by_audio.py            # 按音频裁剪
│   ├── Limb_Segment.txt               # 音频切割结果
│   ├── sample_txts/                   # 样本类别划分
│   └── smpl/                          # SMPL人体模型相关
└── .gitignore                         # Git忽略文件
```

## 主要功能
- 多模态数据采集与预处理（关节点、角度等，一定有左右角度）
- 多种深度学习模型结构（含自定义模块与损失）
- 日志与模型权重自动保存
- 支持交叉验证与多折训练

## 依赖环境
TODO

## 快速开始
1. **数据准备**
   - 将原始数据使用 `raw_data_generate/` 中的脚本进行处理，生成关节点、角度等特征。
2. **配置参数**
   - 修改 `aistroke/config/config.py` 以适配路径和参数。
3. **训练模型**
   - 进入 `aistroke/` 目录，运行：
     ```bash
     python main.py
     # 或
     python CT_CV_main.py
     ```
4. **查看日志与模型**
   - 训练日志与模型权重保存在 `aistroke/logs/` 目录下。

## 训练与推理流程
0. 数据采集与预处理（`raw_data_generate/`）
1. 特征生成与增强（`aistroke/data/`）
2. 配置参数（`aistroke/config/`）
3. 模型训练与验证（`aistroke/main.py`、`CT_CV_main.py`）
4. 日志与模型保存（`aistroke/logs/`）
5. 推理与评估（可扩展）
