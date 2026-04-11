# MLETT - 基于ETT数据集的机器学习时间序列预测

一个使用ETT（电力变压器温度）数据集进行时间序列预测的机器学习流水线。

## 项目结构

```
MLETT/
├── data/
│   ├── raw/                        # 原始数据文件
│   └── processed/                  # 处理后的数据文件
├── src/mlett/                      # 源代码（命名空间包）
│   ├── __init__.py                 # 包入口
│   ├── config/                     # 配置
│   │   ├── __init__.py
│   │   └── config.yaml            # 主配置文件
│   ├── data/                       # 数据处理
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── time_series_split.py
│   ├── features/                   # 特征工程
│   │   ├── __init__.py
│   │   └── engineering.py
│   ├── models/                     # 机器学习模型
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── xgboost_model.py
│   ├── training/                   # 训练和评估
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       ├── io.py
│       ├── logger.py
│       └── metrics.py
├── scripts/                        # 可执行脚本
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   └── predict.py                  # 预测脚本
├── models/                         # 保存的模型
├── logs/                           # 日志文件
├── results/                        # 实验结果
├── notebook/                       # Jupyter笔记本
├── pyproject.toml                  # 包配置（PEP 621）
├── requirements.txt                # 依赖
├── .gitignore
├── README.md
└── README_CN.md

```

## 安装

### 前置要求
- Python 3.8 或更高版本
- Conda 环境

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

> 项目使用 `pyproject.toml`（PEP 621规范）进行包配置。`-e` 参数以开发模式安装，代码修改后即时生效。

## 使用方法

### 训练模型

使用默认配置训练模型：
```bash
python scripts/train.py
```

使用自定义模型名称训练：
```bash
python scripts/train.py --model-name my_model
```

使用自定义配置：
```bash
python scripts/train.py --config path/to/config.yaml
```

### 模型评估

评估训练好的模型：
```bash
python scripts/evaluate.py --model models/model_x.pkl --data data/processed/test_data.csv
```

### 模型预测

使用训练好的模型进行预测：
```bash
python scripts/predict.py --model models/model_x.pkl --data data/processed/input.csv --output predictions.csv
```

## 配置

主配置文件是 `src/mlett/config/config.yaml`。您可以自定义：
- 数据路径和分割比例
- 特征工程选项
- 模型参数
- 训练设置
- 评估指标

## 数据

本项目使用ETT（电力变压器温度）数据集：
- **特征**：HUFL, HULL, MUFL, MULL, LUFL, LULL（各种负荷类型）
- **目标**：OT（油温）
- **粒度**：小时级数据
- **数据集大小**：17,421 个样本

## 模型

项目目前实现了：
- **XGBoost**：用于回归的梯度提升决策树

## 功能特性

- 时间序列数据分割
- 特征工程（基于时间的特征、滚动窗口、滞后特征）
- 模型训练和评估
- 全面的评估指标（MSE、RMSE、MAE、R²、MAPE）
- 日志和实验跟踪
- 模型持久化

## 开发

### 项目结构

项目遵循模块化结构：
- **数据模块**：数据加载、清洗和分割
- **特征模块**：特征工程和转换
- **模型模块**：机器学习模型实现
- **训练模块**：模型训练和评估逻辑
- **工具模块**：辅助函数（日志、I/O、指标）

### 添加新模型

1. 在 `src/mlett/models/` 中创建新的模型类
2. 继承自 `BaseModel`
3. 实现所需的抽象方法
4. 更新训练器以支持新的模型类型

### 包导入方式

项目使用 `mlett` 命名空间包，所有模块通过以下方式导入：
```python
from mlett.data.preprocessing import load_data, clean_data
from mlett.features.engineering import feature_pipeline
from mlett.training.trainer import Trainer
from mlett.utils.logger import setup_logger
```

## 结果

训练结果保存在：
- `models/`：训练好的模型文件
- `results/`：评估指标和配置
- `logs/`：训练和执行日志

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎贡献！请随时提交 Pull Request。