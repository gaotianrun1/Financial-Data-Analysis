import numpy as np
import torch

# 导入各个模块
from dataset.data_loader import download_data, prepare_data_x, prepare_data_y, TimeSeriesDataset
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

# 下载数据
data_date, data_close_price, num_data_points, display_date_range = download_data(config)

# 绘制原始数据
plot_raw_data(data_date, data_close_price, config, num_data_points, display_date_range)

# 数据标准化
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

# 准备数据
data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

# 分割数据集
split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

# 准备绘图数据
to_plot_data_y_train = np.zeros(num_data_points)
to_plot_data_y_val = np.zeros(num_data_points)

to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)

to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

# 绘制训练验证数据分割
plot_train_val_split(data_date, to_plot_data_y_train, to_plot_data_y_val, config, num_data_points)

# 创建数据集
dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

# 创建模型
model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

# 训练模型
model = train_model(model, dataset_train, dataset_val, config)

# 进行预测
predicted_train = predict_on_dataset(model, dataset_train, config)
predicted_val = predict_on_dataset(model, dataset_val, config)

# 准备绘图数据
to_plot_data_y_train_pred = np.zeros(num_data_points)
to_plot_data_y_val_pred = np.zeros(num_data_points)

to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

# 绘制预测结果
plot_predictions(data_date, data_close_price, to_plot_data_y_train_pred, to_plot_data_y_val_pred, config, num_data_points)

# 准备放大图的数据
to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
to_plot_predicted_val = scaler.inverse_transform(predicted_val)
to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]

# 绘制验证数据放大图
plot_validation_zoom(to_plot_data_date, to_plot_data_y_val_subset, to_plot_predicted_val, config)

# 预测下一个交易日的收盘价
prediction = predict_next_day(model, data_x_unseen, config)

# 准备绘图
plot_range = 10
to_plot_data_y_val = np.zeros(plot_range)
to_plot_data_y_val_pred = np.zeros(plot_range)
to_plot_data_y_test_pred = np.zeros(plot_range)

to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

to_plot_data_y_test_pred[plot_range-1] = scaler.inverse_transform(prediction)

to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

# 绘制下一日预测
plot_date_test = data_date[-plot_range+1:]
plot_date_test.append("tomorrow")

plot_next_day_prediction(plot_date_test, to_plot_data_y_val, to_plot_data_y_val_pred, to_plot_data_y_test_pred, config)

print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))

# 打印输出文件总结
print_output_summary() 