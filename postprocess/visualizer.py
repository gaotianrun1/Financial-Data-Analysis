import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

# 设置matplotlib后端为非交互式
plt.ioff()  # 关闭交互模式
plt.switch_backend('Agg')  # 使用非GUI后端

def ensure_output_dir():
    """确保输出目录存在"""
    output_dir = "outputs/plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

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

def print_output_summary():
    """打印输出文件总结"""
    output_dir = "outputs/plots"
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        if files:
            print(f"\n=== 图像输出总结 ===")
            print(f"输出目录: {output_dir}")
            print(f"生成的图像文件:")
            for file in sorted(files):
                print(f"  - {file}")
            print(f"总计: {len(files)} 个图像文件")
            print("=" * 20) 