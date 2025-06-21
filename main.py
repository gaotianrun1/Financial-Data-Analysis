import numpy as np
import torch
import pandas as pd
import os
import time
import random

from dataset.data_loader import TimeSeriesDataset
from dataset.preprocess import check_preprocessed_data_exists, load_preprocessed_data, preprocess_data
from model.model_factory import create_model
from train.trainer import train_model
from inference.predictor import predict_on_dataset
from config import CONFIG
from postprocess.evaluator import evaluate_model_performance, save_evaluation_results
from postprocess.visualizer import create_comprehensive_plots
from datetime import datetime
import pickle

def set_random_seed(seed=42, deterministic=True):
    """
    设置所有随机种子以确保实验结果的可重复性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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
    
    # 创建数据集
    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)
    
    config["data"]["window_size"] = metadata["window_size"]
    config["model"]["input_size"] = metadata["feature_count"]
    
    print(f"\n=== 数据配置 ===")
    print(f"输入维度: {config['model']['input_size']}")
    print(f"窗口大小: {config['data']['window_size']}")
    print(f"训练轮数: {config['training']['num_epoch']}")
    print(f"批次大小: {config['training']['batch_size']}")
    
    model_type = config["model"].get("model_type", "lstm")
    print(f"当前模型类型: {model_type.upper()}")
    
    # 创建模型
    model = create_model(config)
    model = model.to(selected_device)
    
    # 创建时间戳文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = f"_{model_type}" if model_type != "lstm" else ""
    output_dir = f"outputs/{timestamp}{model_suffix}"
    print(f"创建输出目录: {output_dir}")

    models_dir = f"{output_dir}/models"
    checkpoints_dir = f"{output_dir}/checkpoints"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # 训练模型
    print(f"\n=== 开始训练 ({model_type.upper()}模型) ===")
    start_time = time.time()
    
    model, training_history = train_model(model, dataset_train, dataset_val, config, checkpoints_dir)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练完成，耗时: {training_time:.2f} 秒")
    
    # 保存训练历史
    if config["training"]["save_history"]:
        history_path = f"{output_dir}/training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(training_history, f)
        print(f"训练历史已保存到: {history_path}")
    
    # 预测结果
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
    # 添加模型信息到评估结果
    evaluation_results['model_type'] = model_type
    evaluation_results['model_config'] = config["model"]
    evaluation_results['training_time'] = training_time
    
    save_evaluation_results(evaluation_results, output_dir)

    # 保存预测结果
    try:
        results_df = pd.DataFrame({
            'actual_train': data_y_train_original,
            'predicted_train': predicted_train_original
        })
        train_results_path = f"{output_dir}/train_predictions_{model_type}.csv"
        results_df.to_csv(train_results_path, index=False)
        
        results_val_df = pd.DataFrame({
            'actual_val': data_y_val_original,
            'predicted_val': predicted_val_original
        })
        val_results_path = f"{output_dir}/val_predictions_{model_type}.csv"
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
    model_path = f"{models_dir}/{model_type}_model_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存到: {model_path}")

if __name__ == "__main__":
    # python main.py
    set_random_seed()
    main() 

