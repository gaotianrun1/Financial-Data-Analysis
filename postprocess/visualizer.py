import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import pandas as pd
import pickle
import json
import glob
import pdb

plt.ioff()
plt.switch_backend('Agg')

def ensure_output_dir(output_dir="outputs/plots"):
    if not os.path.exists(output_dir): # 如果输出目录不存在，则创建
        os.makedirs(output_dir)
    return output_dir

def plot_prediction_comparison(y_true, y_pred, output_dir, dataset_name="validation", 
                             n_samples=1000, title_suffix="", figsize=(30, 10), timestamps=None):
    """
    绘制预测值与真实值的时间曲线对比图
    """
    try:
        output_dir = ensure_output_dir(output_dir)
        
        # 限制显示的样本数量
        n_plot = min(n_samples, len(y_true))
        
        plt.figure(figsize=figsize)
        
        # 转换为相对时间
        timestamps_subset = timestamps[:n_plot]
        timestamps_subset = timestamps_subset.to_pydatetime()
        time_zero = timestamps_subset[0]
        relative_days = [(t.timestamp() - time_zero.timestamp()) / (24 * 3600) for t in timestamps_subset]

        # 统一使用天数显示
        plt.plot(relative_days, y_true[:n_plot], label='Actual', alpha=0.6, linewidth=0.8, color='#2E86AB')
        plt.plot(relative_days, y_pred[:n_plot], label='Predicted', alpha=0.6, linewidth=0.8, color='#E74C3C')
        plt.xlabel('Relative Time (Days)')

        day_step = max(1, int(len(relative_days) // 20))
        tick_positions = relative_days[::day_step]
        plt.xticks(tick_positions, [f'{d:.0f}' for d in tick_positions])
        
        title = f'{dataset_name.capitalize()} Set Prediction Comparison'
        if title_suffix:
            title += f' - {title_suffix}'
        
        plt.title(title)
        plt.ylabel('Target Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"{output_dir}/{dataset_name}_prediction_comparison.png"
        plt.savefig(filename, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"预测对比图已保存到: {filename}")
        
    except Exception as e:
        print(f"生成预测对比图失败: {e}")


def plot_training_history(train_losses, val_losses, output_dir, learning_rates=None, figsize=(15, 5)):
    """
    绘制训练历史曲线
    """
    output_dir = ensure_output_dir(output_dir)
    
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 学习率曲线
    ax2.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2)
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')  # 使用对数刻度显示学习率
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    filename = f"{output_dir}/training_history.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"训练历史图已保存到: {filename}")

def plot_residuals(y_true, y_pred, output_dir, dataset_name="validation", figsize=(12, 5)):
    """
    绘制残差图
    """
    output_dir = ensure_output_dir(output_dir)
    
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 残差散点图
    ax1.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{dataset_name.capitalize()} Set Residuals')
    ax1.grid(True, alpha=0.3)
    
    # 残差直方图
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    filename = f"{output_dir}/{dataset_name}_residuals.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"残差图已保存到: {filename}")

def plot_scatter_comparison(y_true, y_pred, output_dir, dataset_name="validation", figsize=(8, 8)):
    """
    绘制真实值vs预测值的散点图
    """
    output_dir = ensure_output_dir(output_dir)
    
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # 理想预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{dataset_name.capitalize()} Set: True vs Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"{output_dir}/{dataset_name}_scatter_comparison.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"散点对比图已保存到: {filename}")

def create_comprehensive_plots(train_true, train_pred, val_true, val_pred, 
                             output_dir, train_losses=None, val_losses=None, learning_rates=None,
                             train_timestamps=None, val_timestamps=None):
    """
    创建综合的可视化图表
    """
    print("\n=== 生成可视化图表 ===")
    
    plots_dir = os.path.join(output_dir, "plots")
    
    # 限制最大样本数
    max_train_samples = len(train_true) # min(10000, len(train_true)) 
    max_val_samples = len(val_true) # min(10000, len(val_true))      
    
    plot_prediction_comparison(train_true, train_pred, plots_dir, "training", 
                             n_samples=max_train_samples, timestamps=train_timestamps)
    plot_prediction_comparison(val_true, val_pred, plots_dir, "validation", 
                             n_samples=max_val_samples, timestamps=val_timestamps)
    
    plot_scatter_comparison(train_true, train_pred, plots_dir, "training")
    plot_scatter_comparison(val_true, val_pred, plots_dir, "validation")
    
    plot_residuals(train_true, train_pred, plots_dir, "training")
    plot_residuals(val_true, val_pred, plots_dir, "validation")
    
    plot_training_history(train_losses, val_losses, plots_dir, learning_rates)
    
    print("所有可视化图表生成完成")

def load_result_data(result_dir):
    print(f"正在从 {result_dir} 加载结果数据...")
    
    data = {}

    train_pred_files = glob.glob(os.path.join(result_dir, "train_predictions_*.csv"))
    val_pred_files = glob.glob(os.path.join(result_dir, "val_predictions_*.csv"))
    
    if train_pred_files:
        train_pred_file = train_pred_files[0]
        train_df = pd.read_csv(train_pred_file)
        data['train_true'] = train_df['actual_train'].values
        data['train_pred'] = train_df['predicted_train'].values

    if val_pred_files:
        val_pred_file = val_pred_files[0]
        val_df = pd.read_csv(val_pred_file)  
        data['val_true'] = val_df['actual_val'].values
        data['val_pred'] = val_df['predicted_val'].values

    history_file = os.path.join(result_dir, "training_history.pkl")
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
            if 'train_loss' in history and 'val_loss' in history:
                data['train_losses'] = history['train_loss']
                data['val_losses'] = history['val_loss']
                data['learning_rates'] = history.get('learning_rate', None)
    
    eval_file = os.path.join(result_dir, "evaluation_results.json")
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            data['evaluation'] = json.load(f)
    
    if 'train_true' in data:
        train_timestamps = pd.date_range('2000-01-01', periods=len(data['train_true']), freq='T') # 模拟时间戳
        data['train_timestamps'] = train_timestamps
    
    if 'val_true' in data:
        # 接续训练数据之后
        val_start = train_timestamps[-1] + pd.Timedelta(minutes=1) if 'train_timestamps' in data else pd.Timestamp('2024-06-01')
        val_timestamps = pd.date_range(val_start, periods=len(data['val_true']), freq='T')
        data['val_timestamps'] = val_timestamps

    return data


def result_visualizations(result_dir="outputs/result_lstm"):
    """
    直接从结果目录加载数据进行可视化
    """
    print("=== 开始可视化功能 ===")
    data = load_result_data(result_dir)
    
    create_comprehensive_plots(
        data['train_true'], data['train_pred'],
        data['val_true'], data['val_pred'],
        result_dir,
        data.get('train_losses'), data.get('val_losses'), data.get('learning_rates'),
        data.get('train_timestamps'), data.get('val_timestamps')
    )

    print("\n=== 数据统计信息 ===")
    print(f"训练集样本数: {len(data['train_true'])}")
    print(f"验证集样本数: {len(data['val_true'])}")
    if 'train_losses' in data:
        print(f"训练epoch数: {len(data['train_losses'])}")
    
    if 'evaluation' in data:
        print(f"\n=== 评估指标 ===")
        train_metrics = data['evaluation'].get('train', {})
        val_metrics = data['evaluation'].get('validation', {})
        print(f"训练集 RMSE: {train_metrics.get('rmse', 'N/A'):.4f}")
        print(f"验证集 RMSE: {val_metrics.get('rmse', 'N/A'):.4f}")
        print(f"训练集 R²: {train_metrics.get('r2', 'N/A'):.4f}")
        print(f"验证集 R²: {val_metrics.get('r2', 'N/A'):.4f}")


if __name__ == "__main__":
    # python postprocess/visualizer.py

    result_dir = "outputs/result_transformer" # "outputs/result_transformer" # "outputs/result_lstm"
    result_visualizations(result_dir)
