import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
from datetime import datetime

def calculate_metrics(y_true, y_pred, dataset_name=""):
    """
    计算回归任务的评价指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        dataset_name: 数据集名称（用于输出）
    
    Returns:
        dict: 包含各种指标的字典
    """
    try:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算额外指标
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        return metrics
    
    except Exception as e:
        print(f"计算{dataset_name}指标时出错: {e}")
        return None

def print_metrics(metrics, dataset_name=""):
    """
    打印评价指标
    
    Args:
        metrics: 指标字典
        dataset_name: 数据集名称
    """
    if metrics is None:
        print(f"{dataset_name}指标计算失败")
        return
    
    print(f"{dataset_name} - MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, "
          f"RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}, MAPE: {metrics['mape']:.2f}%")

def evaluate_model_performance(train_true, train_pred, val_true, val_pred, test_true=None, test_pred=None):
    """
    全面评估模型性能
    
    Args:
        train_true, train_pred: 训练集真实值和预测值
        val_true, val_pred: 验证集真实值和预测值
        test_true, test_pred: 测试集真实值和预测值（可选）
    
    Returns:
        dict: 完整的评估结果
    """
    print("\n=== 模型性能评估 ===")
    
    results = {}
    
    # 训练集评估
    train_metrics = calculate_metrics(train_true, train_pred, "训练集")
    if train_metrics:
        results['train'] = train_metrics
        print_metrics(train_metrics, "训练集")
    
    # 验证集评估
    val_metrics = calculate_metrics(val_true, val_pred, "验证集")
    if val_metrics:
        results['validation'] = val_metrics
        print_metrics(val_metrics, "验证集")
    
    # 测试集评估
    if test_true is not None and test_pred is not None:
        test_metrics = calculate_metrics(test_true, test_pred, "测试集")
        if test_metrics:
            results['test'] = test_metrics
            print_metrics(test_metrics, "测试集")
    
    return results

def save_evaluation_results(results, output_dir, filename="evaluation_results.json"):
    """
    保存评估结果到JSON文件
    
    Args:
        results: 评估结果字典
        output_dir: 输出目录
        filename: 文件名
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # 添加时间戳
        results['evaluation_timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {filepath}")
        
    except Exception as e:
        print(f"保存评估结果失败: {e}")
