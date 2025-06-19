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
from postprocess.evaluator import evaluate_model_performance, save_evaluation_results
from postprocess.visualizer import create_comprehensive_plots
from datetime import datetime

def select_device():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return "cuda:0"
    else:
        print("CUDA不可用")
        return "cpu"

def main():
    DATA_DIR = "data"

    config = CONFIG.copy()

    selected_device = select_device()
    config["training"]["device"] = selected_device

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
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)
    
    config["data"]["window_size"] = metadata["window_size"]
    config["model"]["input_size"] = metadata["feature_count"]
    
    print(f"\n=== 模型配置 ===")
    print(f"输入维度: {config['model']['input_size']}")
    print(f"窗口大小: {config['data']['window_size']}")
    print(f"训练轮数: {config['training']['num_epoch']}")
    print(f"批次大小: {config['training']['batch_size']}")
    
    # 创建模型
    model = LSTMModel(
        input_size=config["model"]["input_size"], 
        hidden_layer_size=config["model"]["lstm_size"],  # 减少隐藏层大小以加快训练
        num_layers=config["model"]["num_lstm_layers"],          # 减少LSTM层数
        output_size=1, 
        dropout=config["model"]["dropout"]            # 减少dropout
    )
    model = model.to(selected_device)
    print(f"模型已创建并移动到设备: {selected_device}")
    print(f"模型参数: 隐藏层大小={config['model']['lstm_size']}, "
          f"LSTM层数={config['model']['num_lstm_layers']}, "
          f"dropout={config['model']['dropout']}")
    
    # 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{timestamp}"
    print(f"创建输出目录: {output_dir}")

    models_dir = f"{output_dir}/models"
    checkpoints_dir = f"{output_dir}/checkpoints"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 训练模型
    print("\n=== 开始训练 ===")
    start_time = time.time()
    
    model, training_history = train_model(model, dataset_train, dataset_val, config, checkpoints_dir)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练完成，耗时: {training_time:.2f} 秒")
    
    # 保存训练历史
    if config["training"]["save_history"]:
        import pickle
        history_path = f"{output_dir}/training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(training_history, f)
        print(f"训练历史已保存到: {history_path}")
    
    # 进行预测
    predicted_train = predict_on_dataset(model, dataset_train, config)
    predicted_val = predict_on_dataset(model, dataset_val, config)
    
    # 反标准化预测结果
    predicted_train_original = target_scaler.inverse_transform(predicted_train.reshape(-1, 1)).flatten()
    predicted_val_original = target_scaler.inverse_transform(predicted_val.reshape(-1, 1)).flatten()
    data_y_train_original = target_scaler.inverse_transform(data_y_train.reshape(-1, 1)).flatten()
    data_y_val_original = target_scaler.inverse_transform(data_y_val.reshape(-1, 1)).flatten()
    
    # 计算预测指标
    evaluation_results = evaluate_model_performance(
        data_y_train_original, predicted_train_original,
        data_y_val_original, predicted_val_original
    )
    save_evaluation_results(evaluation_results, output_dir)

    # 保存预测结果
    try:
        results_df = pd.DataFrame({
            'actual_train': data_y_train_original,
            'predicted_train': predicted_train_original
        })
        train_results_path = f"{output_dir}/train_predictions_final.csv"
        results_df.to_csv(train_results_path, index=False)
        
        results_val_df = pd.DataFrame({
            'actual_val': data_y_val_original,
            'predicted_val': predicted_val_original
        })
        val_results_path = f"{output_dir}/val_predictions_final.csv"
        results_val_df.to_csv(val_results_path, index=False)
        
        print(f"预测结果已保存到: {train_results_path}, {val_results_path}")
    except Exception as e:
        print(f"保存预测结果出错: {e}")
    
    create_comprehensive_plots(
        data_y_train_original, predicted_train_original,
        data_y_val_original, predicted_val_original,
        output_dir,
        training_history.get('train_loss', None), 
        training_history.get('val_loss', None),
        training_history.get('learning_rate', None)
    )

    # 保存模型
    model_path = f"{models_dir}/lstm_model_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存到: {model_path}")

if __name__ == "__main__":
    main() 

