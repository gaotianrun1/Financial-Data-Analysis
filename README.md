# 时间序列预测 PyTorch 项目

一个基于 PyTorch 的端到端时间序列预测框架，支持多种深度学习模型和自动化的特征工程。特别针对金融量化交易场景设计。项目提供了从数据预处理、特征工程、模型训练到结果可视化的完整工作流程。

## 安装和环境配置

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
- **TimeAwareTransformer**：在Transformer基础上引入了更强的时间感知


### 3. 结果分析

```bash
# 训练结果可视化
python postprocess/visualizer.py

# 回测
python postprocess/backtester.py
```

## 📊 输出结果

程序运行后会在 `outputs/` 目录下生成带时间戳的结果文件夹
