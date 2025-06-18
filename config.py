CONFIG = {
    "alpha_vantage": {
        "key": "1WJUXBNJDN3V4CKQ", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        # 数据源设置
        "data_source": "parquet",  # "alpha_vantage" 或 "parquet"
        "train_path": "/ssdwork/gaotianrun/findataset/train.parquet",
        "test_path": "/ssdwork/gaotianrun/findataset/test.parquet",
        "processed_dir": "outputs/processed_data",
        
        # 特征设置
        "target_column": "label",
        "feature_selection": "all",  # "all", "top_k", 或指定特征列表
        "top_k_features": 50,  # 当feature_selection="top_k"时使用
        
        # 时间序列设置
        "window_size": 20,
        "train_split_size": 0.80,
        
        # 数据预处理设置
        "handle_outliers": True,
        "outlier_method": "iqr",  # "iqr", "zscore"
        "add_time_features": True,
    },
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 896, # 将根据实际选择的特征数量动态调整
        "num_lstm_layers": 2,
        "lstm_size": 64,  # 增加隐藏层大小以处理更多特征
        "dropout": 0.3,   # 增加dropout以防止过拟合
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 10,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
} 