import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

plt.ioff()
plt.switch_backend('Agg')

def ensure_output_dir(output_dir="outputs/plots"):
    if not os.path.exists(output_dir): # 如果输出目录不存在，则创建
        os.makedirs(output_dir)
    return output_dir

def plot_prediction_comparison(y_true, y_pred, output_dir, dataset_name="validation", 
                             n_samples=200, title_suffix="", figsize=(12, 6)):
    """
    绘制预测值与真实值的时间or序号曲线对比图
    """
    try:
        output_dir = ensure_output_dir(output_dir)
        
        # 限制显示的样本数量
        n_plot = min(n_samples, len(y_true))
        
        plt.figure(figsize=figsize)
        plt.plot(y_true[:n_plot], label='Actual', alpha=0.7, linewidth=1.5)
        plt.plot(y_pred[:n_plot], label='Predicted', alpha=0.7, linewidth=1.5)
        
        title = f'{dataset_name.capitalize()} Set Prediction Comparison'
        if title_suffix:
            title += f' - {title_suffix}'
        
        plt.title(title)
        plt.xlabel('Sample Index')
        plt.ylabel('Target Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"{output_dir}/{dataset_name}_prediction_comparison.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"预测对比图已保存到: {filename}")
        
    except Exception as e:
        print(f"生成预测对比图失败: {e}")

def plot_training_history(train_losses, val_losses, output_dir, figsize=(10, 6)):
    """
    绘制训练历史曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        output_dir: 输出目录
        figsize: 图像大小
    """
    try:
        output_dir = ensure_output_dir(output_dir)
        
        plt.figure(figsize=figsize)
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        filename = f"{output_dir}/training_history.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"训练历史图已保存到: {filename}")
        
    except Exception as e:
        print(f"生成训练历史图失败: {e}")

def plot_residuals(y_true, y_pred, output_dir, dataset_name="validation", figsize=(12, 5)):
    """
    绘制残差图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        output_dir: 输出目录
        dataset_name: 数据集名称
        figsize: 图像大小
    """
    try:
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
        
        # 保存图像
        filename = f"{output_dir}/{dataset_name}_residuals.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"残差图已保存到: {filename}")
        
    except Exception as e:
        print(f"生成残差图失败: {e}")

def plot_scatter_comparison(y_true, y_pred, output_dir, dataset_name="validation", figsize=(8, 8)):
    """
    绘制真实值vs预测值的散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        output_dir: 输出目录
        dataset_name: 数据集名称
        figsize: 图像大小
    """
    try:
        output_dir = ensure_output_dir(output_dir)
        
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # 绘制理想预测线 (y=x)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{dataset_name.capitalize()} Set: True vs Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        filename = f"{output_dir}/{dataset_name}_scatter_comparison.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"散点对比图已保存到: {filename}")
        
    except Exception as e:
        print(f"生成散点对比图失败: {e}")

def create_comprehensive_plots(train_true, train_pred, val_true, val_pred, 
                             output_dir, train_losses=None, val_losses=None):
    """
    创建综合的可视化图表
    
    Args:
        train_true, train_pred: 训练集真实值和预测值
        val_true, val_pred: 验证集真实值和预测值
        output_dir: 输出目录
        train_losses, val_losses: 训练和验证损失（可选）
    """
    print("\n=== 生成可视化图表 ===")
    
    plots_dir = os.path.join(output_dir, "plots")
    
    # 预测对比图
    plot_prediction_comparison(train_true, train_pred, plots_dir, "training", n_samples=200)
    plot_prediction_comparison(val_true, val_pred, plots_dir, "validation", n_samples=200)
    
    # 散点对比图
    plot_scatter_comparison(train_true, train_pred, plots_dir, "training")
    plot_scatter_comparison(val_true, val_pred, plots_dir, "validation")
    
    # 残差图
    plot_residuals(train_true, train_pred, plots_dir, "training")
    plot_residuals(val_true, val_pred, plots_dir, "validation")
    
    # 训练历史图（如果提供）
    if train_losses is not None and val_losses is not None:
        plot_training_history(train_losses, val_losses, plots_dir)
    
    print("所有可视化图表生成完成")

# 保留原有的函数以兼容旧代码
def plot_raw_data(data_date, data_close_price, config, num_data_points, display_date_range):
    output_dir = ensure_output_dir()
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    
    # 保存图像
    filename = f"{output_dir}/01_raw_data_{config['alpha_vantage']['symbol']}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()  # 关闭图像以释放内存
    print(f"图像已保存: {filename}")

def plot_train_val_split(data_date, to_plot_data_y_train, to_plot_data_y_val, config, num_data_points):
    output_dir = ensure_output_dir()
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
    plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] + " - showing training and validation data")
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    plt.legend()
    
    # 保存图像
    filename = f"{output_dir}/02_train_val_split_{config['alpha_vantage']['symbol']}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()  # 关闭图像以释放内存
    print(f"图像已保存: {filename}")

def plot_predictions(data_date, data_close_price, to_plot_data_y_train_pred, to_plot_data_y_val_pred, config, num_data_points):
    output_dir = ensure_output_dir()
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
    plt.title("Compare predicted prices to actual prices")
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    plt.legend()
    
    # 保存图像
    filename = f"{output_dir}/03_predictions_comparison_{config['alpha_vantage']['symbol']}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()  # 关闭图像以释放内存
    print(f"图像已保存: {filename}")

def plot_validation_zoom(to_plot_data_date, to_plot_data_y_val_subset, to_plot_predicted_val, config):
    output_dir = ensure_output_dir()
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
    plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
    plt.title("Zoom in to examine predicted price on validation data portion")
    xticks = [to_plot_data_date[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(to_plot_data_date)-i) > config["plots"]["xticks_interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))] # make x ticks nice
    xs = np.arange(0,len(xticks))
    plt.xticks(xs, xticks, rotation='vertical')
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    plt.legend()
    
    # 保存图像
    filename = f"{output_dir}/04_validation_zoom_{config['alpha_vantage']['symbol']}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()  # 关闭图像以释放内存
    print(f"图像已保存: {filename}")

def plot_next_day_prediction(plot_date_test, to_plot_data_y_val, to_plot_data_y_val_pred, to_plot_data_y_test_pred, config):
    output_dir = ensure_output_dir()
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
    plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
    plt.title("Predicting the close price of the next trading day")
    plt.grid(visible=True, which='major', axis='y', linestyle='--')
    plt.legend()
    
    # 保存图像
    filename = f"{output_dir}/05_next_day_prediction_{config['alpha_vantage']['symbol']}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()  # 关闭图像以释放内存
    print(f"图像已保存: {filename}")
