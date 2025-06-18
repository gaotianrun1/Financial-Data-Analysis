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
from config import CONFIG

print("All libraries loaded")

# 使用配置文件
config = CONFIG.copy()

# 快速测试配置 - 大幅减少数据量和训练时间
config["data"]["feature_selection"] = "top_k"
config["data"]["top_k_features"] = 10  # 只使用前10个最相关的特征
config["data"]["window_size"] = 10     # 减少窗口大小
config["training"]["num_epoch"] = 2    # 只训练2轮
config["training"]["batch_size"] = 32  # 减少batch size

# 快速测试参数
SAMPLE_SIZE = 10000  # 只使用前10000个样本进行测试
print(f"=== 快速测试配置 ===")
print(f"样本数量限制: {SAMPLE_SIZE}")
print(f"特征数量: {config['data']['top_k_features']}")
print(f"窗口大小: {config['data']['window_size']}")
print(f"训练轮数: {config['training']['num_epoch']}")
print(f"批次大小: {config['training']['batch_size']}")


print("=== 加载数据 ===")
train_df, test_df = load_processed_data(config, use_cache=True)

# 限制数据量用于快速测试
print(f"原始训练数据大小: {train_df.shape}")
train_df_sample = train_df.head(SAMPLE_SIZE)
print(f"采样后训练数据大小: {train_df_sample.shape}")

# 确定特征列
all_feature_cols = [col for col in train_df_sample.columns if col != config["data"]["target_column"]]
target_col = config["data"]["target_column"]

# 特征选择 - 使用与目标变量最相关的前K个特征
print("=== 进行特征选择 ===")
correlations = train_df_sample[all_feature_cols].corrwith(train_df_sample[target_col]).abs()
top_features = correlations.nlargest(config["data"]["top_k_features"]).index.tolist()
feature_cols = top_features

print(f"使用的特征数量: {len(feature_cols)}")
print(f"选择的特征: {feature_cols}")
print(f"目标列: {target_col}")

# 更新模型配置中的输入维度
config["model"]["input_size"] = len(feature_cols)
print(f"更新模型输入维度为: {config['model']['input_size']}")

# 准备时间序列数据
print("=== 准备时间序列数据 ===")
data_x, data_x_unseen = prepare_multivariate_data_x(train_df_sample, config["data"]["window_size"], feature_cols)
data_y = prepare_multivariate_data_y(train_df_sample, config["data"]["window_size"], target_col)

print(f"输入数据形状: {data_x.shape}")  # (n_samples, window_size, n_features)
print(f"目标数据形状: {data_y.shape}")  # (n_samples,)
print(f"未见数据形状: {data_x_unseen.shape}")  # (window_size, n_features)

print("=== 数据形状解释 ===")
print(f"  - 第一维 {data_x.shape[0]}: 样本数量（时间序列窗口数）")
print(f"  - 第二维 {data_x.shape[1]}: 时间步长（窗口大小）")
print(f"  - 第三维 {data_x.shape[2]}: 特征维度数")

# 数据标准化
print("=== 数据标准化 ===")
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
    hidden_layer_size=32,  # 减少隐藏层大小以加快训练
    num_layers=1,          # 减少LSTM层数
    output_size=1, 
    dropout=0.2            # 减少dropout
)
model = model.to(config["training"]["device"])
print(f"模型已创建并移动到设备: {config['training']['device']}")
print(f"模型参数: 隐藏层大小=32, LSTM层数=1, dropout=0.2")

# 训练模型
print("=== 开始训练 ===")
import time
start_time = time.time()

model = train_model(model, dataset_train, dataset_val, config)

end_time = time.time()
training_time = end_time - start_time
print(f"训练完成，耗时: {training_time:.2f} 秒")

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
try:
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
except Exception as e:
    print(f"评估指标计算出错: {e}")

# 预测下一个时间点
print("=== 预测下一个时间点 ===")
try:
    prediction_normalized = predict_next_day(model, data_x_unseen_normalized, config)
    prediction_original = target_scaler.inverse_transform(prediction_normalized.reshape(-1, 1)).flatten()[0]

    print(f"下一个时间点的预测值: {prediction_original:.6f}")
except Exception as e:
    print(f"下一个时间点预测出错: {e}")

# 保存模型和结果
print("=== 保存结果 ===")
import os
os.makedirs("outputs/models", exist_ok=True)
torch.save(model.state_dict(), "outputs/models/lstm_model_fast_test.pth")
print("模型已保存到: outputs/models/lstm_model_fast_test.pth")

# 保存预测结果
try:
    results_df = pd.DataFrame({
        'actual_train': data_y_train_original,
        'predicted_train': predicted_train_original
    })
    results_df.to_csv("outputs/train_predictions_fast_test.csv", index=False)

    results_val_df = pd.DataFrame({
        'actual_val': data_y_val_original,
        'predicted_val': predicted_val_original
    })
    results_val_df.to_csv("outputs/val_predictions_fast_test.csv", index=False)

    print("预测结果已保存到 outputs/ 目录")
except Exception as e:
    print(f"保存预测结果出错: {e}")

print("=== 快速测试完成 ===")
print(f"总耗时: {training_time:.2f} 秒")
print(f"使用了 {len(feature_cols)} 个特征进行训练")
print(f"训练数据: {data_x_train.shape[0]} 样本")
print(f"验证数据: {data_x_val.shape[0]} 样本")
print(f"数据形状: {data_x.shape} (样本数, 时间步, 特征数)")

# 简单的可视化预测结果
try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    
    # 绘制验证集的预测结果（只显示前100个点）
    n_plot = min(100, len(data_y_val_original))
    
    plt.figure(figsize=(12, 6))
    plt.plot(data_y_val_original[:n_plot], label='实际值', alpha=0.7)
    plt.plot(predicted_val_original[:n_plot], label='预测值', alpha=0.7)
    plt.title('验证集预测结果对比（前100个样本）')
    plt.xlabel('样本索引')
    plt.ylabel('目标值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/fast_test_prediction_comparison.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    print("预测对比图已保存到: outputs/plots/fast_test_prediction_comparison.png")
    
except Exception as e:
    print(f"生成可视化图表失败: {e}") 