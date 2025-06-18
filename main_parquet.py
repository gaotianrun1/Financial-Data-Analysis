import numpy as np
import torch
import pandas as pd

# 导入各个模块
from dataset.data_loader import (
    load_processed_data, 
    prepare_multivariate_data_x, 
    prepare_multivariate_data_y,
    TimeSeriesDataset
)
from utils.normalizer import Normalizer
from model.lstm_model import LSTMModel
from train.trainer import train_model
from inference.predictor import predict_on_dataset, predict_next_day
from postprocess.visualizer import (
    plot_raw_data, 
    plot_train_val_split, 
    plot_predictions, 
    plot_validation_zoom, 
    plot_next_day_prediction,
    print_output_summary
)
from config import CONFIG

print("All libraries loaded")

# 使用配置文件
config = CONFIG

# 检查数据源设置
if config["data"]["data_source"] != "parquet":
    print("警告：配置文件中的数据源不是parquet，请检查config.py中的data_source设置")

# 加载预处理后的数据
print("=== 加载数据 ===")
train_df, test_df = load_processed_data(config, use_cache=True)

# 确定特征列
all_feature_cols = [col for col in train_df.columns if col != config["data"]["target_column"]]
target_col = config["data"]["target_column"]

# 特征选择
if config["data"]["feature_selection"] == "all":
    feature_cols = all_feature_cols
elif config["data"]["feature_selection"] == "top_k":
    # 基于与目标变量的相关性选择top_k特征
    correlations = train_df[all_feature_cols].corrwith(train_df[target_col]).abs()
    top_features = correlations.nlargest(config["data"]["top_k_features"]).index.tolist()
    feature_cols = top_features
    print(f"选择了前{len(feature_cols)}个最相关的特征")
else:
    # 自定义特征列表
    feature_cols = config["data"]["feature_selection"]

print(f"使用的特征数量: {len(feature_cols)}")
print(f"目标列: {target_col}")

# 更新模型配置中的输入维度
config["model"]["input_size"] = len(feature_cols)
print(f"更新模型输入维度为: {config['model']['input_size']}")

# 准备时间序列数据
print("=== 准备时间序列数据 ===")
data_x, data_x_unseen = prepare_multivariate_data_x(train_df, config["data"]["window_size"], feature_cols)
data_y = prepare_multivariate_data_y(train_df, config["data"]["window_size"], target_col)

print(f"输入数据形状: {data_x.shape}")  # (n_samples, window_size, n_features)
print(f"目标数据形状: {data_y.shape}")  # (n_samples,)
print(f"未见数据形状: {data_x_unseen.shape}")  # (window_size, n_features)

# 数据标准化
print("=== 数据标准化 ===")
# 分别标准化特征和目标
feature_scaler = Normalizer()
target_scaler = Normalizer()

# 重塑数据以进行标准化
original_shape = data_x.shape
data_x_reshaped = data_x.reshape(-1, data_x.shape[-1])  # (n_samples * window_size, n_features)
data_x_normalized_reshaped = feature_scaler.fit_transform(data_x_reshaped)
data_x_normalized = data_x_normalized_reshaped.reshape(original_shape)

# 标准化目标数据
data_y_normalized = target_scaler.fit_transform(data_y.reshape(-1, 1)).flatten()

# 标准化未见数据
data_x_unseen_normalized = feature_scaler.transform(data_x_unseen.reshape(-1, data_x_unseen.shape[-1])).reshape(data_x_unseen.shape)

print("数据标准化完成")

# 分割数据集
print("=== 分割数据集 ===")
split_index = int(data_y_normalized.shape[0] * config["data"]["train_split_size"])
data_x_train = data_x_normalized[:split_index]
data_x_val = data_x_normalized[split_index:]
data_y_train = data_y_normalized[:split_index]
data_y_val = data_y_normalized[split_index:]

print(f"训练集大小: {data_x_train.shape[0]}")
print(f"验证集大小: {data_x_val.shape[0]}")

# 创建数据集
dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

# 创建模型
print("=== 创建模型 ===")
model = LSTMModel(
    input_size=config["model"]["input_size"], 
    hidden_layer_size=config["model"]["lstm_size"],
    num_layers=config["model"]["num_lstm_layers"], 
    output_size=1, 
    dropout=config["model"]["dropout"]
)
model = model.to(config["training"]["device"])
print(f"模型已创建并移动到设备: {config['training']['device']}")

# 训练模型
print("=== 开始训练 ===")
model = train_model(model, dataset_train, dataset_val, config)

# 进行预测
print("=== 进行预测 ===")
predicted_train = predict_on_dataset(model, dataset_train, config)
predicted_val = predict_on_dataset(model, dataset_val, config)

# 反标准化预测结果
predicted_train_original = target_scaler.inverse_transform(predicted_train.reshape(-1, 1)).flatten()
predicted_val_original = target_scaler.inverse_transform(predicted_val.reshape(-1, 1)).flatten()
data_y_train_original = target_scaler.inverse_transform(data_y_train.reshape(-1, 1)).flatten()
data_y_val_original = target_scaler.inverse_transform(data_y_val.reshape(-1, 1)).flatten()

# 计算预测指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=== 预测性能评估 ===")
# 训练集性能
train_mse = mean_squared_error(data_y_train_original, predicted_train_original)
train_mae = mean_absolute_error(data_y_train_original, predicted_train_original)
train_r2 = r2_score(data_y_train_original, predicted_train_original)

print(f"训练集 - MSE: {train_mse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")

# 验证集性能
val_mse = mean_squared_error(data_y_val_original, predicted_val_original)
val_mae = mean_absolute_error(data_y_val_original, predicted_val_original)
val_r2 = r2_score(data_y_val_original, predicted_val_original)

print(f"验证集 - MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")

# 预测下一个时间点
print("=== 预测下一个时间点 ===")
prediction_normalized = predict_next_day(model, data_x_unseen_normalized, config)
prediction_original = target_scaler.inverse_transform(prediction_normalized.reshape(-1, 1)).flatten()[0]

print(f"下一个时间点的预测值: {prediction_original:.6f}")

# 保存模型和结果
print("=== 保存结果 ===")
import os
os.makedirs("outputs/models", exist_ok=True)
torch.save(model.state_dict(), "outputs/models/lstm_model_parquet.pth")
print("模型已保存到: outputs/models/lstm_model_parquet.pth")

# 保存预测结果
results_df = pd.DataFrame({
    'actual_train': data_y_train_original,
    'predicted_train': predicted_train_original
})
results_df.to_csv("outputs/train_predictions.csv", index=False)

results_val_df = pd.DataFrame({
    'actual_val': data_y_val_original,
    'predicted_val': predicted_val_original
})
results_val_df.to_csv("outputs/val_predictions.csv", index=False)

print("预测结果已保存到 outputs/ 目录")

print("=== 程序执行完成 ===")
print_output_summary() 