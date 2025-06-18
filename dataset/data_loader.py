import numpy as np
import pandas as pd
import os
from alpha_vantage.timeseries import TimeSeries
from torch.utils.data import Dataset

def load_parquet_data(config):
    """
    加载本地parquet数据文件，进行数据清洗和预处理
    """
    print("Loading data from parquet files...")
    
    # 读取数据
    train_df = pd.read_parquet(config["data"]["train_path"])
    test_df = pd.read_parquet(config["data"]["test_path"])

    # 数据清洗和预处理
    train_cleaned = process_data(train_df)
    test_cleaned = process_data(test_df)
    
    # 保存预处理后的parquet数据
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    train_processed_path = os.path.join(processed_dir, "train_processed.parquet")
    test_processed_path = os.path.join(processed_dir, "test_processed.parquet")
    
    train_cleaned.to_parquet(train_processed_path)
    test_cleaned.to_parquet(test_processed_path)
    
    return train_cleaned, test_cleaned

def process_data(df):
    """
    数据清洗，处理缺失值和极端异常值
    """
    df_processed = df.copy()
    
    # 检查和处理缺失值
    missing_count = df_processed.isnull().sum().sum()
    if missing_count > 0:
        print(f"发现 {missing_count} 个缺失值，使用前向填充处理")
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    # 简化的异常值处理（只处理极端异常值）
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'label']  # 排除标签列
    
    # 只处理极端异常值（使用更宽松的3倍标准差）
    outlier_counts = {}
    for col in numeric_cols:
        # 跳过标准差为0的列（常数列）
        if df_processed[col].std() == 0:
            continue
            
        mean_val = df_processed[col].mean()
        std_val = df_processed[col].std()
        
        # 使用3倍标准差作为异常值阈值（更保守）
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        
        outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            outlier_counts[col] = outlier_count
            # 使用边界值替换异常值
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
    
    if outlier_counts:
        print(f"处理了 {len(outlier_counts)} 个特征的极端异常值")
        total_outliers = sum(outlier_counts.values())
        print(f"总共处理了 {total_outliers} 个极端异常值")
    else:
        print("未发现需要处理的极端异常值")
    
    # # 特征工程：添加时间相关特征（如果有时间戳索引）
    # if isinstance(df_processed.index, pd.DatetimeIndex):
    #     df_processed['hour'] = df_processed.index.hour
    #     df_processed['minute'] = df_processed.index.minute
    #     df_processed['day_of_week'] = df_processed.index.dayofweek
    #     print("添加了时间相关特征: hour, minute, day_of_week")
    
    # 检查并处理可能的无穷值
    inf_count = np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"发现 {inf_count} 个无穷值，将其替换为0")
        df_processed = df_processed.replace([np.inf, -np.inf], 0)
    
    return df_processed

def load_processed_data(config, use_cache=True):
    """
    加载清理后的parquet数据，用于在preprocess中保存为pickle
    """
    processed_dir = config["data"]["processed_dir"]
    train_processed_path = os.path.join(processed_dir, "train_processed.parquet")
    test_processed_path = os.path.join(processed_dir, "test_processed.parquet")
    
    # 检查是否存在预处理后的文件
    if use_cache and os.path.exists(train_processed_path) and os.path.exists(test_processed_path):
        print("加载预处理后的缓存数据...")
        train_df = pd.read_parquet(train_processed_path)
        test_df = pd.read_parquet(test_processed_path)
        return train_df, test_df
    else:
        print("缓存不存在或选择重新处理，开始加载原始数据...")
        return load_parquet_data(config)

# def download_data(config):
#     """
#     保留原有的Alpha Vantage API下载功能（向后兼容）
#     """
#     ts = TimeSeries(key=config["alpha_vantage"]["key"])
#     # 使用免费的 get_daily 而不是付费的 get_daily_adjusted
#     data, meta_data = ts.get_daily(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

#     data_date = [date for date in data.keys()]
#     data_date.reverse()

#     # 使用普通的收盘价而不是调整后的收盘价
#     data_close_price = [float(data[date]["4. close"]) for date in data.keys()]
#     data_close_price.reverse()
#     data_close_price = np.array(data_close_price)

#     num_data_points = len(data_date)
#     display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
#     print("Number data points", num_data_points, display_date_range)

#     return data_date, data_close_price, num_data_points, display_date_range

def prepare_data_x(df, window_size, feature_cols):
    """
    Args:
        df: pandas DataFrame，包含时间序列数据
        window_size: 窗口大小
        feature_cols: 特征列名列表
    
    Returns:
        data_x: 形状为 (n_samples, window_size, n_features) 的数组
        data_x_unseen: 用于预测下一个时间点的最后一个窗口
    """
    # 提取特征数据
    feature_data = df[feature_cols].values.astype(np.float32)
    
    n_samples = feature_data.shape[0] - window_size + 1
    n_features = len(feature_cols)
    
    # 创建滑动窗口
    data_x = np.zeros((n_samples, window_size, n_features))
    
    for i in range(n_samples):
        data_x[i] = feature_data[i:i+window_size]

    # 最后一个窗口没有可预测值
    data_x_unseen = feature_data[-window_size:]
    
    return data_x[:-1], data_x_unseen

def prepare_data_y(df, window_size, target_col='label'):
    target_data = df[target_col].values.astype(np.float32)
    data_y = target_data[window_size:]
    return data_y

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        if len(x.shape) == 2:
            # [batch, sequence] -> [batch, sequence, features]
            x = np.expand_dims(x, 2)
        elif len(x.shape) == 3:
            # [batch, sequence, features]
            pass
        else:
            raise ValueError(f"输入数据x的维度不正确，期望2,3，得到{len(x.shape)}D")
            
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
        print(f"Dataset created with input shape: {self.x.shape}, target shape: {self.y.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx]) 