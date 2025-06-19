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
        "processed_dir": "data/processed_data",

        # 数据采样
        "sample_size": 10000,
        "test_sample_size": 2000,
        
        # 特征设置
        "target_column": "label",
        "feature_selection": "top_k",  # "all", "top_k", 或指定特征列表
        "top_k_features": 50,  # 当feature_selection="top_k"时使用
        
        # 时间序列设置
        "window_size": 20,
        "train_split_size": 0.90,
        
        # 数据预处理设置
        "handle_outliers": True,
        "outlier_method": "iqr",  # "iqr", "zscore"
        "add_time_features": True,
    },
    "data_processing": {
        # 缺失值处理
        "missing_threshold": 0.2,        # 缺失比例超过20%的特征将被删除
        "missing_fill_method": "ffill",   # 'ffill', 'bfill', 'interpolate'
        
        # 方差过滤
        "variance_threshold": 1e-4,       # 方差阈值，低于此值的特征将被删除
        
        # 相关性过滤
        "correlation_threshold": 0.95,    # 相关系数阈值，高于此值的特征对将删除其中一个
        
        # 基于标签的特征选择
        "feature_selection_method": "both",  # 'correlation', 'mutual_info', 'both'
        "feature_keep_ratio": 0.3,          # 保留特征的比例（删除70%最没价值的特征）
        
        # 异常值处理
        "outlier_method": "clip",            # 'clip' 截断, 'remove' 删除
        "outlier_std_threshold": 3,          # 标准差阈值
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
        "input_size": 896, # 将根据实际选择的特征数量动态调整，删掉一些特征
        "num_lstm_layers": 2,
        "lstm_size": 64,  # 增加隐藏层大小以处理更多特征
        "dropout": 0.3,   # 增加dropout以防止过拟合
    },
    "training": {
        "device": "cuda", # "cuda" or "cpu"
        "batch_size": 512,
        "num_epoch": 100,
        "learning_rate": 0.001,
        "scheduler_step_size": 40,
        
        # 新增训练优化配置
        "scheduler_type": "cosine",  # "step", "cosine", "plateau"
        "print_interval": 10,         # 每隔多少个epoch打印一次
        "save_history": True,        # 是否保存训练历史
        
        # Checkpoint保存设置
        "checkpoint": {
            "enabled": True,         # 是否启用checkpoint保存
            "save_interval": 15,    # 每隔多少个epoch保存一次
            "max_keep": 0,          # 最多保留多少个checkpoint（0表示保留所有）
        }
    }
} 