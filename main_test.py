import numpy as np
import torch
import pandas as pd
import os
import time

from dataset.data_loader import TimeSeriesDataset
from dataset.preprocess import check_preprocessed_data_exists, load_preprocessed_data, preprocess_data
from model.lstm_model import LSTMModel
from train.trainer import train_model
from inference.predictor import predict_on_dataset
from config import CONFIG
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

def select_device():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU设备")

        selected_device = "cuda:0"

        torch.cuda.empty_cache()

        return selected_device
    else:
        print("CUDA不可用")
        return "cpu"

def main():
    FEATURE_COUNT = 10
    WINDOW_SIZE = 10
    DATA_DIR = "data"
    
    print("=" * 60)
    print("快速测试配置")
    print("=" * 60)
    print(f"特征数量: {FEATURE_COUNT}")
    print(f"窗口大小: {WINDOW_SIZE}")
    print(f"数据目录: {DATA_DIR}")

    config = CONFIG.copy()

    selected_device = select_device()
    config["training"]["device"] = selected_device
    
    print("\n=== 检查预处理数据 ===")
    if not check_preprocessed_data_exists(DATA_DIR):
        print("未找到预处理数据!")
    else:
        print("找到预处理数据，正在加载...")
    
    # 加载预处理数据
    try:
        data_dict = load_preprocessed_data(DATA_DIR)
        train_data = data_dict['train_data']
        val_data = data_dict['val_data']
        test_data = data_dict['test_data']
        scalers = data_dict['scalers']
        metadata = data_dict['metadata']

        data_x_train = train_data['x']
        data_y_train = train_data['y']
        data_x_val = val_data['x']
        data_y_val = val_data['y']
        data_x_test = test_data['x']
        data_y_test = test_data['y']
        
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']
        
        print("数据加载成功!")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 创建数据集
    print("\n=== 创建数据集 ===")
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)
    
    print(f"训练集大小: {len(dataset_train)}")
    print(f"验证集大小: {len(dataset_val)}")
    
    # 快速测试配置
    config["data"]["window_size"] = metadata["window_size"]
    config["model"]["input_size"] = metadata["feature_count"]
    config["training"]["num_epoch"] = 20
    config["training"]["batch_size"] = 32
    
    print(f"\n=== 模型配置 ===")
    print(f"输入维度: {config['model']['input_size']}")
    print(f"窗口大小: {config['data']['window_size']}")
    print(f"训练轮数: {config['training']['num_epoch']}")
    print(f"批次大小: {config['training']['batch_size']}")
    
    # 创建模型
    print("\n=== 创建模型 ===")
    model = LSTMModel(
        input_size=config["model"]["input_size"], 
        hidden_layer_size=32,  # 减少隐藏层大小以加快训练
        num_layers=1,          # 减少LSTM层数
        output_size=1, 
        dropout=0.2            # 减少dropout
    )
    model = model.to(selected_device)
    print(f"模型已创建并移动到设备: {selected_device}")
    print(f"模型参数: 隐藏层大小=32, LSTM层数=1, dropout=0.2")
    
    # 训练模型
    print("\n=== 开始训练 ===")
    start_time = time.time()
    
    model = train_model(model, dataset_train, dataset_val, config)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练完成，耗时: {training_time:.2f} 秒")
    
    # 进行预测
    print("\n=== 进行预测 ===")
    predicted_train = predict_on_dataset(model, dataset_train, config)
    predicted_val = predict_on_dataset(model, dataset_val, config)
    
    # 反标准化预测结果
    predicted_train_original = target_scaler.inverse_transform(predicted_train.reshape(-1, 1)).flatten()
    predicted_val_original = target_scaler.inverse_transform(predicted_val.reshape(-1, 1)).flatten()
    data_y_train_original = target_scaler.inverse_transform(data_y_train.reshape(-1, 1)).flatten()
    data_y_val_original = target_scaler.inverse_transform(data_y_val.reshape(-1, 1)).flatten()
    
    # 计算预测指标
    print("\n=== 预测性能评估 ===")
    try:  
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

    # 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{timestamp}"
    models_dir = f"{output_dir}/models"
    plots_dir = f"{output_dir}/plots"
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"创建输出目录: {output_dir}")
    
    # 保存模型
    model_path = f"{models_dir}/lstm_model_fast_test.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存预测结果
    try:
        results_df = pd.DataFrame({
            'actual_train': data_y_train_original,
            'predicted_train': predicted_train_original
        })
        train_results_path = f"{output_dir}/train_predictions_fast_test.csv"
        results_df.to_csv(train_results_path, index=False)
        
        results_val_df = pd.DataFrame({
            'actual_val': data_y_val_original,
            'predicted_val': predicted_val_original
        })
        val_results_path = f"{output_dir}/val_predictions_fast_test.csv"
        results_val_df.to_csv(val_results_path, index=False)
        
        print(f"预测结果已保存到: {train_results_path}, {val_results_path}")
    except Exception as e:
        print(f"保存预测结果出错: {e}")
    
    # 可视化
    try:
        plt.switch_backend('Agg')
        
        # 绘制验证集的预测结果（只显示前100个点）
        n_plot = min(200, len(data_y_val_original))
        
        plt.figure(figsize=(12, 6))
        plt.plot(data_y_val_original[:n_plot], label='Actual', alpha=0.7)
        plt.plot(predicted_val_original[:n_plot], label='Predicted', alpha=0.7)
        plt.title('Validation Set Prediction Comparison (First 100 Samples)')
        plt.xlabel('Sample Index')
        plt.ylabel('Target Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = f"{plots_dir}/prediction_comparison.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"预测对比图已保存到: {plot_path}")
        
    except Exception as e:
        print(f"生成可视化图表失败: {e}")
    
    print(f"训练完成，总耗时: {training_time:.2f} 秒")
    print(f"训练数据形状: {data_x_train.shape} (样本数, 时间步, 特征数)")

if __name__ == "__main__":
    main() 

