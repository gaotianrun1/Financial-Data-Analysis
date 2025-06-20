CONFIG = {
    "alpha_vantage": {
        "key": "1WJUXBNJDN3V4CKQ", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
    },
    "data": {
        "data_source": "parquet",  # "alpha_vantage" 或 "parquet"
        "train_path": "/ssdwork/gaotianrun/findataset/train.parquet",
        "test_path": "/ssdwork/gaotianrun/findataset/test_converted.parquet",
        "processed_dir": "data/processed_data",

        # 数据采用
        "sample_size": 100000,
        "test_sample_size": 43200,
        
        # 特征设置
        "target_column": "label",
        
        # 时间序列设置
        "window_size": 60, # 被preprocess覆盖掉了
        "train_split_size": 0.80,
    },
    "data_processing": {
        # 缺失值处理
        "missing_threshold": 0.2,        # 缺失比例超过20%的特征将被删除
        "missing_fill_method": "ffill",   # 'ffill', 'bfill', 'interpolate'
        
        # 方差过滤
        "variance_threshold": 1e-4,       # 方差阈值，低于此值的特征将被删除
        
        # 相关性过滤
        "correlation_threshold": 0.95,    # 相关系数阈值，高于此值的特征对将删除其中一个
        "correlation_sample_size": 10000, # 计算相关性矩阵时的采样数量
        
        # 基于标签的特征选择
        "feature_selection_method": "both",  # 'correlation', 'mutual_info', 'both'
        "feature_keep_ratio": 0.3,          # 保留特征的比例（删除70%最没价值的特征）
        "mutual_info_sample_size": 20000,   # 计算互信息时的采样数量
        
        # 数据异常值处理
        "outlier_method": "iqr",             # 'iqr', 'quantile', 'zscore'
        "outlier_window": "1D",              # 时间窗口大小 "1D", "1H", "15min", "5min"
        "protected_features": ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"]
    },
    # 量化金融特征工程配置
    "feature_engineering": {
        "enable_feature_engineering": True,
        "enable_order_flow_features": True,      # 启用订单流特征
        "enable_liquidity_features": True,       # 启用流动性特征
        "enable_microstructure_features": True,  # 启用微观结构特征
        "enable_pressure_features": True,        # 启用买卖压力特征
        "enable_statistical_features": True,     # 启用统计特征
        "enable_time_features": True,           # 启用时间特征
        "enable_interaction_features": True,     # 启用交互特征
        
        # 订单流特征配置
        "order_flow_windows": [5, 15, 30],       # 订单流滚动窗口
        
        # 流动性特征配置
        "liquidity_windows": [5, 15, 30],        # 流动性特征滚动窗口
        
        # 微观结构特征配置
        "microstructure_windows": [5, 15, 30],   # 微观结构特征滚动窗口
        # 买卖压力特征配置
        "pressure_windows": [5, 15, 30],         # 压力特征滚动窗口
        
        # 统计特征配置
        "statistical_windows": [5, 15, 30],      # 统计特征窗口
        "lag_periods": [1, 2, 3, 5, 10],        # 滞后期
        "rolling_operations": ["mean", "std", "max", "min"], # 滚动统计操作
        
        # 时间特征配置
        "enable_cyclical_encoding": True,        # 启用周期性编码
        
        # 交互特征配置
        "enable_time_interactions": True,        # 启用时间交互特征
        "enable_volume_interactions": True,      # 启用成交量交互特征
        "enable_pressure_interactions": True,    # 启用压力交互特征
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
        # 通用模型设置
        "model_type": "lstm",  # "lstm" 或 "transformer"
        "input_size": 896, # 将根据实际选择的特征数量动态调整，删掉一些特征
        "output_size": 1,
        "dropout": 0.3,   # 增加dropout以防止过拟合
        
        # LSTM特定设置
        "num_lstm_layers": 2,
        "lstm_size": 64,  # 增加隐藏层大小以处理更多特征
        
        # Transformer特定设置
        "num_transformer_layers": 2,
        "transformer_hidden_size": 64,  # 必须能被attention_heads整除
        "num_attention_heads": 4,       # 注意力头数量
        "transformer_dropout": 0.15,
    },
    "training": {
        "device": "cuda", # "cuda" or "cpu"
        "batch_size": 2048,
        "num_epoch": 30,
        "learning_rate": 0.001,
        
        # 训练优化配置
        "scheduler_type": "cosine",  # "step", "cosine", "plateau"
        "print_interval": 1,         # 每隔多少个epoch打印一次
        "save_history": True,        # 是否保存训练历史
        "scheduler_step_size": 40,
        
        # Checkpoint保存设置
        "checkpoint": {
            "enabled": True,         # 是否启用checkpoint保存
            "save_interval": 150,    # 每隔多少个epoch保存一次
            "max_keep": 0,          # 最多保留多少个checkpoint（0表示保留所有）
        }
    }
} 