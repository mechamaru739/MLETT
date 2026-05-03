# MLETT - 基于ETT数据集的机器学习时间序列预测

**[English README](README.md)**

一个使用ETT（电力变压器温度）数据集进行时间序列预测的机器学习流水线。

## 项目结构

```
MLETT/
├── data/
│   ├── raw/                            # 原始数据文件
│   │   └── ETTh1.csv
│   └── processed/                      # 处理后的数据文件
├── src/mlett/                          # 源代码（命名空间包）
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config.yaml                # 主配置文件
│   │   └── tune_config.yaml           # 超参数调优配置
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py            # 数据加载与清洗
│   │   └── time_series_split.py        # 按时间分割与滑动窗口
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py             # FeatureTransformer（防数据泄露）
│   │   └── industrial_features.py     # 紧凑42维工业特征
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py              # 抽象基类
│   │   ├── xgboost_model.py           # XGBoost模型
│   │   └── lightgbm_model.py          # LightGBM模型
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                 # 训练器（含反标准化）
│   └── utils/
│       ├── __init__.py
│       ├── io.py                       # I/O工具（YAML、JSON、CSV）
│       ├── logger.py                   # 日志设置
│       ├── metrics.py                  # MSE、RMSE、MAE、R2、MAPE、SMAPE
│       └── seed.py                     # 随机种子（可复现性）
├── scripts/
│   ├── train.py                        # 训练脚本
│   ├── evaluate.py                     # 评估脚本
│   ├── predict.py                      # 预测脚本
│   └── tune.py                         # 超参数调优（Grid/Optuna）
├── results/                            # 实验输出（每次运行一个目录）
├── pyproject.toml                      # 包配置（PEP 621）
├── .gitignore
├── README.md
└── README_CN.md
```

## 安装

### 前置要求
- Python 3.8+
- Conda

### 安装步骤

1. 创建 conda 环境：
```bash
conda create -n data_science python=3.9
conda activate data_science
```

2. 安装包（包含所有依赖）：
```bash
pip install -e .
```

> 使用 `pyproject.toml`（PEP 621规范）。`-e` 参数以开发模式安装，代码修改即时生效。

## 使用方法

### 训练

```bash
# 默认配置（XGBoost + 工业特征）
python scripts/train.py

# 自定义实验名称
python scripts/train.py --model-name my_experiment

# 自定义配置文件
python scripts/train.py --config path/to/config.yaml
```

### 评估

```bash
python scripts/evaluate.py --experiment results/<实验名称> --data data/raw/ETTh1.csv
```

### 预测

```bash
python scripts/predict.py --experiment results/<实验名称> --input data/raw/new_data.csv
```

### 超参数调优

```bash
# 网格搜索
python scripts/tune.py --tune-config src/mlett/config/tune_config.yaml

# Optuna 贝叶斯优化（tune_config.yaml 中默认）
python scripts/tune.py --tune-config src/mlett/config/tune_config.yaml
```

结果保存到 `results/tune_<时间戳>/`，包含 `tune_comparison.yaml`（最佳验证指标和参数）。

## 训练流水线

训练流水线严格按顺序执行，防止数据泄露：

```
1. 加载并清洗原始数据
2. 按时间顺序分割（训练70% / 验证15% / 测试15%）— 在特征工程之前
3. FeatureTransformer.fit_transform(训练集)  — 仅在训练集上拟合 scaler 和 target_scaler
   FeatureTransformer.transform(验证集)       — 使用已拟合的 scaler
   FeatureTransformer.transform(测试集)       — 使用已拟合的 scaler
4. 对每个分割分别构建滑动窗口样本
5. 在标准化后的特征和目标上训练模型
6. 反标准化预测值到原始尺度，计算指标
7. 保存所有产物到 results/<实验名称>/ 目录
```

### 特征模式

| 模式 | 维度 | 说明 |
|------|------|------|
| `industrial` | 42 | 滚动统计量(24) + 近期值(12) + 时间特征(6) |
| `flat` | 288 | 将所有(24步 × 12特征)铺平 |

**工业特征**（默认）将每个滑动窗口压缩为紧凑的42维向量：
- **滚动统计量 (24维)**：6个传感器在24h窗口内的 mean/std/max/min
- **近期值 (12维)**：t-1 和 t 时刻的传感器原始值
- **目标点时间特征 (6维)**：hour_sin、hour_cos、month_sin、month_cos、day_of_week、is_weekend

### 目标模式

| 模式 | 说明 |
|------|------|
| `delta` | 预测OT变化量：`OT[t+h] - OT[t+h-1]`（默认） |
| `absolute` | 直接预测OT绝对值 |

Delta 模式降低目标方差，通常能改善泛化性能。指标始终在原始尺度上通过重建后计算。

### 样本定义（滑动窗口）

每个样本的定义：
- **输入 X**：过去 `window_size` 小时的特征（默认：24h）
- **输出 y**：未来 `forecast_horizon` 小时的 OT 值（默认：1h）

```
┌─────────────────────┐    ┌───┐
│  t-23 ... t-1, t    │ →  │t+1│
│  24h × 特征          │    │OT │
└─────────────────────┘    └───┘
```

### 目标标准化

训练时特征和目标（OT）都会被标准化：
- 特征：`StandardScaler` 仅在训练集上拟合
- 目标：独立的 `target_scaler` 仅在训练集上拟合
- 指标在**原始尺度**上计算（反标准化后）
- `target_scaler` 随 `transformer.pkl` 一起保存，推理时自动使用

## 模型

### XGBoost（默认）

基于直方图的梯度提升树。使用 `reg:absoluteerror` 目标函数（MAE损失），对异常值更鲁棒。

### LightGBM

替代梯度提升框架，采用 leaf-wise 树生长、GOSS 采样和 EFB 特征捆绑。通常比 XGBoost 训练更快。

通过 `config.yaml` 切换模型：
```yaml
model:
  type: "lightgbm"  # 或 "xgboost"
```

### 模型对比（ETT数据集，工业特征，delta模式）

| 模型 | 测试RMSE | 测试R² | 测试SMAPE |
|------|----------|--------|-----------|
| XGBoost | 0.6187 | 0.9524 | 6.27% |
| LightGBM | 0.6229 | 0.9518 | 6.31% |

## 超参数调优

通过 `tune_config.yaml` 支持两种调优方法：

### 网格搜索
对参数值进行穷举笛卡尔积。适合小搜索空间。

### Optuna（贝叶斯优化）
TPE 采样器 + MedianPruner 剪枝，提前终止无前景的试验。对大搜索空间效率更高。

搜索空间按模型区分 — `tune_config.yaml` 中包含 XGBoost（`search_space`）和 LightGBM（`lightgbm_search_space`）的独立配置块。

## 实验目录

每次训练运行生成一个自包含的实验目录：

```
results/
└── <实验名称>/                 ← 时间戳或 --model-name 的值
    ├── train.log               ← 主训练日志
    ├── trainer.log             ← Trainer模块日志
    ├── model.pkl               ← 训练好的模型
    ├── model_results.yaml      ← 训练/验证指标
    ├── transformer.pkl          ← FeatureTransformer（含 target_scaler）
    ├── config.yaml              ← 配置快照
    └── results.yaml             ← 测试指标与样本定义
```

调优运行将所有试验保存在带时间戳的目录下：

```
results/
└── tune_<时间戳>/
    ├── tune_comparison.yaml     ← 最佳验证指标和参数
    ├── optuna_trial_000/        ← 每个试验的产物
    ├── optuna_trial_001/
    └── ...
```

## 配置

主配置文件：`src/mlett/config/config.yaml`
调优配置文件：`src/mlett/config/tune_config.yaml`

| 配置节 | 关键设置 |
|--------|---------|
| `data` | raw_data_path、target_column (OT)、numerical_features |
| `split` | 训练/验证/测试比例（0.7/0.15/0.15） |
| `features` | feature_mode、target_mode、window_size、forecast_horizon |
| `model.type` | "xgboost" 或 "lightgbm" |
| `model.xgboost` | objective、max_depth、learning_rate、n_estimators、正则化 |
| `model.lightgbm` | objective、metric、num_leaves、max_depth、learning_rate、正则化 |
| `training` | use_validation、save_model |
| `paths` | results_dir |
| `random_seed` | 42（可复现性） |

## 数据

ETT（电力变压器温度）数据集：
- **特征**：HUFL, HULL, MUFL, MULL, LUFL, LULL（各种负荷类型）
- **目标**：OT（油温）
- **粒度**：小时级
- **大小**：17,421 个样本（2016-07 至 2018-06）

## 功能特性

- 按时间顺序分割数据（不随机打乱，保持时间顺序）
- 防数据泄露的特征工程（仅在训练集上 fit，在验证/测试集上 transform）
- 工业特征提取（紧凑42维表示）
- Delta目标模式（预测温度变化量，而非绝对值）
- 多模型支持（XGBoost、LightGBM）
- 超参数调优（网格搜索 + Optuna 带剪枝）
- 目标标准化与反标准化，指标在原始尺度上计算
- 全面评估指标：MSE、RMSE、MAE、R²、MAPE、SMAPE
- 随机种子保证可复现性（Python、NumPy、XGBoost、LightGBM）
- 自包含的实验目录

## 包导入方式

```python
from mlett.data.preprocessing import load_data, clean_data
from mlett.data.time_series_split import time_series_split, create_sliding_windows
from mlett.features.engineering import FeatureTransformer
from mlett.features.industrial_features import create_industrial_windows
from mlett.training.trainer import Trainer
from mlett.utils.logger import setup_logger
from mlett.utils.seed import set_random_seed
```

## 许可证

MIT 许可证
