# Time Series Forecasting PyTorch - 项目结构文档

## 1. 项目架构概览

这是一个基于PyTorch的时间序列预测项目，主要用于金融时间序列数据的预测。项目采用模块化设计，将数据处理、模型定义、训练、推理和后处理分别封装在不同的模块中。

### 技术栈
- **深度学习框架**: PyTorch
- **数据处理**: pandas, numpy
- **可视化**: matplotlib
- **数据源**: Alpha Vantage API, Parquet文件
- **模型架构**: LSTM神经网络
- **评估指标**: MSE, MAE, RMSE, R², MAPE

## 2. 主要目录结构及其职责

```
time-series-forecasting-pytorch/
├── config.py                 # 全局配置文件
├── requirements.txt           # 项目依赖
├── main.py                   # 主要执行入口（预处理数据）
├── main_test.py              # 快速测试入口
├── main_online_data.py       # 在线数据获取入口
├── main\ copy.py             # 备份版本
├── .gitignore                # Git忽略文件
│
├── dataset/                  # 数据处理模块
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载和清洗
│   ├── preprocess.py         # 数据预处理主模块
│   └── data_preview.py       # 数据预览工具
│
├── model/                    # 模型定义模块
│   ├── __init__.py
│   └── lstm_model.py         # LSTM模型定义
│
├── train/                    # 训练模块
│   ├── __init__.py
│   └── trainer.py            # 训练器实现
│
├── inference/                # 推理模块
│   ├── __init__.py
│   └── predictor.py          # 预测器实现
│
├── postprocess/              # 后处理模块
│   ├── __init__.py
│   ├── evaluator.py          # 模型评估
│   └── visualizer.py         # 可视化图表生成
│
├── utils/                    # 工具模块
│   ├── __init__.py
│   └── normalizer.py         # 数据标准化工具
│
├── data/                     # 数据存储目录
│   ├── train_data.pkl        # 预处理训练数据
│   ├── val_data.pkl          # 预处理验证数据
│   ├── test_data.pkl         # 预处理测试数据
│   ├── scalers.pkl           # 标准化器
│   ├── metadata.pkl          # 元数据信息
│   └── processed_data/       # 处理后的数据缓存
│
└── outputs/                  # 输出结果目录
    └── {timestamp}/          # 按时间戳分组的输出
        ├── models/           # 保存的模型权重
        ├── plots/            # 生成的图表
        ├── *.csv            # 预测结果CSV
        └── evaluation_results.json  # 评估结果
```

## 3. 关键模块的依赖关系图

项目模块间的依赖关系如上图所示，主要流程为：
1. `config.py` 提供全局配置
2. `dataset/` 模块负责数据加载和预处理
3. `model/` 模块定义LSTM模型
4. `train/` 模块执行模型训练
5. `inference/` 模块进行预测推理
6. `postprocess/` 模块处理结果评估和可视化
7. `utils/` 模块提供通用工具函数

## 4. 核心类和接口的功能说明

### 4.1 数据处理类

#### `TimeSeriesDataset` (dataset/data_loader.py)
```python
class TimeSeriesDataset(Dataset)
```
- **功能**: PyTorch数据集包装器，用于时间序列数据
- **输入**: 特征数据(x)和目标数据(y)
- **输出**: 符合PyTorch训练要求的数据格式
- **关键方法**:
  - `__init__(x, y)`: 初始化数据集
  - `__len__()`: 返回数据集大小
  - `__getitem__(idx)`: 获取单个样本

#### `Normalizer` (utils/normalizer.py)
```python
class Normalizer
```
- **功能**: 数据标准化工具，实现Z-score标准化
- **关键方法**:
  - `fit_transform(x)`: 训练标准化器并转换数据
  - `transform(x)`: 使用已训练的标准化器转换数据
  - `inverse_transform(x)`: 反标准化，恢复原始数据尺度

### 4.2 模型类

#### `LSTMModel` (model/lstm_model.py)
```python
class LSTMModel(nn.Module)
```
- **功能**: LSTM时间序列预测模型
- **架构**: 线性层 → ReLU → LSTM → Dropout → 线性输出层
- **参数**:
  - `input_size`: 输入特征维度
  - `hidden_layer_size`: LSTM隐藏层大小
  - `num_layers`: LSTM层数
  - `output_size`: 输出维度
  - `dropout`: Dropout比率
- **关键方法**:
  - `forward(x)`: 前向传播
  - `init_weights()`: 权重初始化

### 4.3 核心函数接口

#### 数据预处理接口
```python
# dataset/preprocess.py
def preprocess_data(sample_size, feature_count, window_size, test_sample_size, save_dir) -> dict
def check_preprocessed_data_exists(data_dir) -> bool
def load_preprocessed_data(data_dir) -> dict
```

#### 训练接口
```python
# train/trainer.py
def train_model(model, dataset_train, dataset_val, config) -> (model, training_history)
def run_epoch(model, dataloader, criterion, optimizer, config, is_training) -> loss
def create_scheduler(optimizer, config) -> scheduler
class EarlyStopping  # 早停功能类
```

#### 推理接口
```python
# inference/predictor.py
def predict_on_dataset(model, dataset, config) -> np.ndarray
```

#### 评估接口
```python
# postprocess/evaluator.py
def evaluate_model_performance(train_true, train_pred, val_true, val_pred, test_true, test_pred) -> dict
def calculate_metrics(y_true, y_pred, dataset_name) -> dict
def save_evaluation_results(results, output_dir, filename)
```

#### 可视化接口
```python
# postprocess/visualizer.py
def create_comprehensive_plots(train_true, train_pred, val_true, val_pred, output_dir, train_losses, val_losses)
def plot_prediction_comparison(y_true, y_pred, output_dir, dataset_name, n_samples, title_suffix, figsize)
def plot_residuals(y_true, y_pred, output_dir, dataset_name, figsize)
def plot_scatter_comparison(y_true, y_pred, output_dir, dataset_name, figsize)
```

## 5. 数据流向图

数据在项目中的流向如上图所示，主要包括：
1. **数据加载阶段**: 从Parquet文件或Alpha Vantage API加载原始数据
2. **数据预处理阶段**: 清洗、特征选择、时间序列窗口化、标准化
3. **训练阶段**: 使用LSTM模型进行训练
4. **推理阶段**: 使用训练好的模型进行预测
5. **后处理阶段**: 评估模型性能和生成可视化结果

## 6. API接口清单

### 6.1 数据处理API
| 函数名 | 模块 | 功能描述 | 输入参数 | 返回值 |
|--------|------|----------|----------|--------|
| `load_parquet_data` | dataset.data_loader | 加载Parquet数据文件 | config | train_df, test_df |
| `process_data` | dataset.data_loader | 主数据预处理函数（集成所有步骤） | df, config, is_training_data | 处理后的DataFrame |
| `handle_missing_values` | dataset.data_loader | 统一处理缺失值、无穷值和删除高缺失率特征 | df, missing_threshold, method | df_processed, removed_features |
| `remove_low_variance_features` | dataset.data_loader | 删除低方差特征 | df, variance_threshold, target_col | df_filtered, removed_features |
| `remove_highly_correlated_features` | dataset.data_loader | 删除高度相关特征 | df, correlation_threshold, target_col | df_filtered, removed_features |
| `select_features_by_target_correlation` | dataset.data_loader | 基于与目标变量相关性选择特征 | df, target_col, keep_ratio, method | df_selected, selected_features, scores |
| `handle_outliers` | dataset.data_loader | 处理异常值 | df, method, std_threshold, target_col | df_processed, outlier_info |
| `prepare_data_x` | dataset.data_loader | 准备时间序列输入数据 | df, window_size, feature_cols | data_x, data_x_unseen |
| `prepare_data_y` | dataset.data_loader | 准备时间序列目标数据 | df, window_size, target_col | data_y |

### 6.2 模型训练API
| 函数名 | 模块 | 功能描述 | 输入参数 | 返回值 |
|--------|------|----------|----------|--------|
| `train_model` | train.trainer | 训练LSTM模型（含早停、历史记录） | model, dataset_train, dataset_val, config | (训练后的模型, 训练历史) |
| `run_epoch` | train.trainer | 执行单个训练轮次 | model, dataloader, criterion, optimizer, config, is_training | 平均损失 |
| `create_scheduler` | train.trainer | 创建学习率调度器 | optimizer, config | 调度器对象 |

### 6.3 推理预测API
| 函数名 | 模块 | 功能描述 | 输入参数 | 返回值 |
|--------|------|----------|----------|--------|
| `predict_on_dataset` | inference.predictor | 在数据集上进行预测 | model, dataset, config | 预测结果数组 |

### 6.4 评估和可视化API
| 函数名 | 模块 | 功能描述 | 输入参数 | 返回值 |
|--------|------|----------|----------|--------|
| `evaluate_model_performance` | postprocess.evaluator | 全面评估模型性能 | 真实值和预测值 | 评估结果字典 |
| `calculate_metrics` | postprocess.evaluator | 计算回归指标 | y_true, y_pred, dataset_name | 指标字典 |
| `create_comprehensive_plots` | postprocess.visualizer | 创建综合可视化图表 | 数据和输出目录 | None (生成图表文件) |
| `plot_prediction_comparison` | postprocess.visualizer | 绘制预测对比图 | y_true, y_pred, output_dir等 | None (生成图表文件) |

## 7. 常见的代码模式和约定

### 7.1 配置管理模式
- 所有配置统一在 `config.py` 中管理
- 使用字典结构分类组织配置项
- 支持运行时动态修改配置

### 7.2 数据处理模式
- 使用Pipeline模式处理数据流
- 数据标准化使用fit-transform模式
- 支持数据缓存机制避免重复处理

### 7.3 模型训练模式
- 使用strategy模式分离训练和验证逻辑
- 支持设备自动检测(CUDA/CPU)
- 实现多种学习率调度器（余弦退火、阶梯、自适应）
- 集成早停机制防止过拟合
- 完整的训练历史记录和可视化
- 智能打印频率控制

### 7.4 结果输出模式
- 使用时间戳创建独立的输出目录
- 结构化保存模型、预测结果和评估报告
- 自动生成多种可视化图表

### 7.5 错误处理模式
- 关键函数使用try-catch异常处理
- 提供详细的错误信息和调试输出
- 优雅处理数据缺失和格式错误

### 7.6 代码风格约定
- 使用中文注释和变量名（适应项目特点）
- 函数参数使用类型提示
- 遵循PEP8代码风格标准
- 使用docstring文档化关键函数

## 8. 项目使用流程

### 8.1 数据预处理流程
```bash
# 直接使用预处理模块
python -m dataset.preprocess
```

### 8.2 完整训练流程
```bash
# 使用主入口（需要预处理数据）
python main.py

# 快速测试（简化配置）
python main_test.py

# 在线数据获取和训练
python main_online_data.py
```

### 8.3 输出结果说明
- **models/**: 保存的模型权重文件(.pth)
- **plots/**: 预测对比图、残差图、散点图、训练历史图等
- ***.csv**: 训练、验证集的预测结果
- **evaluation_results.json**: 详细的评估指标报告
- **training_history.pkl**: 完整的训练历史数据（损失、学习率等）

## 9. 训练优化功能详解

### 9.1 早停机制 (Early Stopping)
- **功能**: 监控验证损失，在指定轮次内无改善时自动停止训练
- **配置参数**:
  - `patience`: 容忍验证损失不改善的最大轮次
  - `min_delta`: 最小改善阈值
  - `restore_best_weights`: 是否恢复最佳权重
- **优势**: 防止过拟合，节省训练时间

### 9.2 学习率调度器
支持三种调度器类型：

#### 余弦退火调度器 (Cosine Annealing)
```python
config["training"]["scheduler_type"] = "cosine"
```
- 学习率按余弦函数平滑下降
- 自动根据总epoch数确定下降过程
- 最小学习率为初始学习率的1%

#### 阶梯调度器 (Step LR)
```python
config["training"]["scheduler_type"] = "step"
```
- 每隔指定轮次降低学习率
- 使用`scheduler_step_size`参数控制

#### 自适应调度器 (Plateau)
```python
config["training"]["scheduler_type"] = "plateau"
```
- 根据验证损失自动调整学习率
- 验证损失停止改善时降低学习率

### 9.3 训练历史记录
训练过程中自动记录：
- 每个epoch的训练损失
- 每个epoch的验证损失
- 学习率变化曲线
- 最佳验证损失及对应epoch
- 是否触发早停等元信息

### 9.4 可视化增强
新增训练历史可视化：
- 训练/验证损失对比图
- 学习率变化曲线（对数刻度）
- 多种调度器效果对比

### 9.5 配置示例
```python
config["training"] = {
    "scheduler_type": "cosine",
    "print_interval": 5,
    "early_stopping": {
        "enabled": True,
        "patience": 15,
        "min_delta": 1e-6,
        "restore_best_weights": True
    },
    "save_history": True
}
```

## 10. 数据预处理优化功能详解

### 10.1 模块化设计
数据预处理模块采用模块化设计，每个功能独立封装：

#### 缺失值和无穷值处理 (`handle_missing_values`)
- **功能**: 统一处理缺失值和无穷值，删除高缺失率特征，填充剩余缺失值
- **参数**: 
  - `missing_threshold`: 缺失比例阈值（默认0.2）
  - `method`: 填充方法（'ffill', 'bfill', 'interpolate'）
- **策略**: 
  1. 将无穷值转换为NaN，与缺失值统一处理
  2. 删除缺失率>20%的特征（包括无穷值）
  3. 对剩余缺失值进行填充
  4. 最终检查确保没有遗留的无穷值

#### 方差过滤 (`remove_low_variance_features`)
- **功能**: 删除方差过小的特征
- **参数**: `variance_threshold`: 方差阈值（默认1e-6）
- **理论**: 低方差特征对模型贡献较小，删除可减少过拟合

#### 相关性过滤 (`remove_highly_correlated_features`)
- **功能**: 删除高度相关的特征对中的一个
- **参数**: `correlation_threshold`: 相关系数阈值（默认0.95）
- **策略**: 计算特征相关矩阵，删除高度相关特征对中的后一个

#### 基于目标的特征选择 (`select_features_by_target_correlation`)
- **功能**: 基于与目标变量的相关性选择特征
- **方法**: 
  - `correlation`: 线性相关系数
  - `mutual_info`: 互信息（捕捉非线性关系）
  - `both`: 两者结合（加权平均）
- **参数**: `keep_ratio`: 保留特征比例（默认0.7，删除30%最没价值的特征）

#### 异常值处理 (`handle_outliers`)
- **功能**: 检测和处理统计意义上的异常值
- **方法**: 
  - `clip`: 截断到边界值
  - `remove`: 删除异常值行
- **参数**: `std_threshold`: 标准差阈值（默认3）
- **注意**: 无穷值已在缺失值处理阶段统一处理

### 10.2 配置参数说明
```python
"data_processing": {
    "missing_threshold": 0.2,        # 缺失比例阈值
    "variance_threshold": 1e-6,      # 方差阈值
    "correlation_threshold": 0.95,   # 相关系数阈值
    "feature_selection_method": "both",  # 特征选择方法
    "feature_keep_ratio": 0.7,       # 保留特征比例
    "outlier_method": "clip",        # 异常值处理方法
    "outlier_std_threshold": 3       # 异常值标准差阈值
}
```

### 10.3 处理流程
1. **缺失值和无穷值处理**: 将无穷值转为NaN统一处理 → 删除高缺失率特征 → 填充剩余缺失值
2. **方差过滤**: 删除常数或近似常数特征
3. **相关性过滤**: 删除高度相关的冗余特征
4. **目标相关性选择**: 基于与标签的相关性选择最有价值的特征
5. **异常值处理**: 处理统计意义上的极端异常值

### 10.4 量化交易特性支持
- 支持量化交易特有的特征（bid_qty, ask_qty, volume等）
- 针对高频数据的特征工程优化
- 考虑时间序列特性的预处理策略

这个项目结构清晰，模块化程度高，便于维护和扩展。每个模块都有明确的职责分工，接口设计简洁易用。训练优化功能和数据预处理优化使得整个机器学习流程更加智能和高效。 