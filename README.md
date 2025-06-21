# 时间序列预测 PyTorch 项目

一个基于 PyTorch 的端到端时间序列预测框架，支持多种深度学习模型和自动化的特征工程。

## 🎯 项目概述

本项目是一个完整的时间序列预测解决方案，特别针对金融量化交易场景设计。项目提供了从数据预处理、特征工程、模型训练到结果可视化的完整工作流程。

## 📁 项目结构

```
time-series-forecasting-pytorch/
├── main.py                    # 主程序入口
├── config.py                  # 配置文件
├── requirements.txt           # 依赖包列表
├── dataset/                   # 数据处理模块
│   ├── data_loader.py        # 数据加载和基础处理
│   ├── preprocess.py         # 数据预处理主流程
│   ├── feature_engineering.py # 特征工程
│   └── data_preview.py       # 数据预览和分析
├── model/                     # 模型定义
│   ├── lstm_model.py         # LSTM 模型
│   ├── transformer_model.py  # Transformer 模型
│   └── model_factory.py      # 模型工厂
├── train/                     # 训练模块
│   └── trainer.py            # 模型训练器
├── inference/                 # 推理模块
│   └── predictor.py          # 预测器
├── postprocess/              # 后处理模块
│   ├── evaluator.py          # 模型评估
│   ├── visualizer.py         # 结果可视化
│   └── backtester.py         # 回测分析
├── utils/                     # 工具模块
│   └── normalizer.py         # 数据标准化
├── data/                      # 数据目录
└── outputs/                   # 输出结果目录
```

## 🛠️ 安装和环境配置

```bash
pip install -r requirements.txt
```

## 🚀 使用指南

### 1. 数据预处理

首先进行数据预处理和特征工程：

```bash
# 启用特征工程的预处理
python -m dataset.preprocess --enable_feature_engineering

# 不启用特征工程的预处理
python -m dataset.preprocess
```

### 2. 模型训练和预测

```bash
# 使用默认配置运行完整流程
python main.py
```

支持的模型类型：
- **LSTM**: 长短期记忆网络，适合序列建模
- **Transformer**: 注意力机制模型，适合长序列依赖建模

### 3. 结果分析

```bash
# 训练结果可视化
python postprocess/visualizer.py

# 回测
python postprocess/backtester.py
```

## 📊 输出结果

程序运行后会在 `outputs/` 目录下生成带时间戳的结果文件夹
