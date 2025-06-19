import numpy as np
import pandas as pd
import os
from alpha_vantage.timeseries import TimeSeries
from torch.utils.data import Dataset
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_parquet_data(config):
    """
    加载本地parquet数据文件，进行数据清洗和预处理
    """
    print("Loading data from parquet files...")
    
    # 读取数据
    train_df = pd.read_parquet(config["data"]["train_path"])
    test_df = pd.read_parquet(config["data"]["test_path"])

    # 数据采前sample_size个样本
    train_df_sample = train_df.head(config["data"]["sample_size"])
    test_df_sample = test_df.head(config["data"]["test_sample_size"])

    # 数据清洗和预处理
    train_cleaned = process_data(train_df_sample, config)
    test_cleaned = process_data(test_df_sample, config, is_training_data=False)
    
    # 保存预处理后的parquet数据
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    train_processed_path = os.path.join(processed_dir, "train_processed.parquet")
    test_processed_path = os.path.join(processed_dir, "test_processed.parquet")
    
    train_cleaned.to_parquet(train_processed_path)
    test_cleaned.to_parquet(test_processed_path)
    
    return train_cleaned, test_cleaned

def handle_missing_values(df, missing_threshold=0.2, method='ffill'):
    """
    处理缺失值和无穷值
    
    Args:
        df: DataFrame
        missing_threshold: 缺失比例阈值，超过此比例的特征将被删除
        method: 填充方法，'ffill'前向填充, 'bfill'后向填充, 'interpolate'插值
    
    Returns:
        df_processed: 处理后的DataFrame
        removed_features: 被删除的特征列表
    """
    print("=== 处理缺失值和无穷值 ===")
    df_processed = df.copy()
    
    # 首先将无穷值转换为NaN，统一处理
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df_processed[numeric_cols]).sum().sum()
    if inf_count > 0:
        print(f"发现 {inf_count} 个无穷值，将其转换为NaN统一处理")
        df_processed[numeric_cols] = df_processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 计算每列的缺失比例（包括原来的NaN和转换后的无穷值）
    missing_ratios = df_processed.isnull().sum() / len(df_processed)
    
    # 识别需要删除的特征（缺失比例过高）
    features_to_remove = missing_ratios[missing_ratios > missing_threshold].index.tolist()
    
    if features_to_remove:
        print(f"删除缺失比例>{missing_threshold:.1%}的特征: {len(features_to_remove)}个")
        print(f"删除的特征: {features_to_remove[:10]}..." if len(features_to_remove) > 10 else f"删除的特征: {features_to_remove}")
        df_processed = df_processed.drop(columns=features_to_remove)
    
    # 处理剩余的缺失值（包括NaN和无穷值）
    remaining_missing = df_processed.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"发现 {remaining_missing} 个缺失值（含无穷值），使用{method}方法填充")
        
        if method == 'ffill':
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        elif method == 'bfill':
            df_processed = df_processed.fillna(method='bfill').fillna(method='ffill')
        elif method == 'interpolate':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].interpolate(method='linear')
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        # 验证是否还有缺失值
        final_missing = df_processed.isnull().sum().sum()
        if final_missing > 0:
            print(f"警告：仍有 {final_missing} 个缺失值，使用0填充")
            df_processed = df_processed.fillna(0)
    
    # 最终检查，确保没有无穷值
    final_inf_count = np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()
    if final_inf_count > 0:
        print(f"警告：仍有 {final_inf_count} 个无穷值，使用0填充")
        df_processed = df_processed.replace([np.inf, -np.inf], 0)
    
    print(f"缺失值和无穷值处理完成，保留特征数: {df_processed.shape[1]}")
    return df_processed, features_to_remove

def remove_low_variance_features(df, variance_threshold=1e-6, target_col='label'):
    """
    删除低方差特征
    
    Args:
        df: DataFrame
        variance_threshold: 方差阈值
        target_col: 目标列名，不参与方差计算
    
    Returns:
        df_filtered: 过滤后的DataFrame
        removed_features: 被删除的特征列表
    """
    print("=== 删除低方差特征 ===")
    
    # 获取数值特征列（排除目标列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        feature_cols = [col for col in numeric_cols if col != target_col]
    else:
        feature_cols = numeric_cols
    
    # 计算方差
    variances = df[feature_cols].var()
    
    # 找出低方差特征
    low_variance_features = variances[variances <= variance_threshold].index.tolist()
    
    if low_variance_features:
        print(f"删除方差<={variance_threshold}的特征: {len(low_variance_features)}个")
        print(f"删除的特征: {low_variance_features[:10]}..." if len(low_variance_features) > 10 else f"删除的特征: {low_variance_features}")
        df_filtered = df.drop(columns=low_variance_features)
    else:
        print("未发现需要删除的低方差特征")
        df_filtered = df.copy()
        
    print(f"方差过滤完成，保留特征数: {df_filtered.shape[1]}")
    return df_filtered, low_variance_features

def remove_highly_correlated_features(df, correlation_threshold=0.95, target_col='label'):
    """
    删除高度相关的特征
    
    Args:
        df: DataFrame
        correlation_threshold: 相关系数阈值
        target_col: 目标列名，不参与相关性计算
    
    Returns:
        df_filtered: 过滤后的DataFrame
        removed_features: 被删除的特征列表
    """
    print("=== 删除高度相关特征 ===")
    
    # 获取特征列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        feature_cols = [col for col in numeric_cols if col != target_col]
    else:
        feature_cols = numeric_cols
    
    # 计算相关矩阵
    print("计算特征相关矩阵...")
    corr_matrix = df[feature_cols].corr().abs()
    
    # 找出高度相关的特征对
    removed_features = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > correlation_threshold:
                # 删除第二个特征（保留第一个）
                feature_to_remove = corr_matrix.columns[j]
                if feature_to_remove not in removed_features:
                    removed_features.append(feature_to_remove)
    
    if removed_features:
        print(f"删除相关系数>{correlation_threshold}的特征: {len(removed_features)}个")
        print(f"删除的特征: {removed_features[:10]}..." if len(removed_features) > 10 else f"删除的特征: {removed_features}")
        df_filtered = df.drop(columns=removed_features)
    else:
        print("未发现需要删除的高度相关特征")
        df_filtered = df.copy()
    
    print(f"相关性过滤完成，保留特征数: {df_filtered.shape[1]}")
    return df_filtered, removed_features

def select_features_by_target_correlation(df, target_col='label', keep_ratio=0.7, method='both'):
    """
    基于与目标变量的相关性选择特征
    
    Args:
        df: DataFrame
        target_col: 目标列名
        keep_ratio: 保留特征的比例
        method: 'correlation' 线性相关, 'mutual_info' 互信息, 'both' 两者结合
    
    Returns:
        df_selected: 特征选择后的DataFrame
        selected_features: 保留的特征列表
        feature_scores: 特征评分字典
    """
    print(f"=== 基于与目标变量相关性的特征选择 (保留{keep_ratio:.1%}) ===")
    
    if target_col not in df.columns:
        print(f"警告：目标列 '{target_col}' 不存在，跳过特征选择")
        return df.copy(), df.columns.tolist(), {}
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    feature_scores = {}
    
    # 计算线性相关性
    if method in ['correlation', 'both']:
        print("计算与目标变量的线性相关性...")
        correlations = X.corrwith(y).abs()
        feature_scores['correlation'] = correlations.to_dict()
    
    # 计算互信息（非线性相关性）
    if method in ['mutual_info', 'both']:
        print("计算与目标变量的互信息...")
        # 对特征进行标准化以提高互信息计算的稳定性
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # 计算互信息，使用更大的样本以提高计算速度
        sample_size = min(30000, len(X_scaled))
        if len(X_scaled) > sample_size:
            sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X_scaled
            y_sample = y
        
        # 计算互信息
        mutual_info_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
        feature_scores['mutual_info'] = dict(zip(feature_cols, mutual_info_scores))
    
    # 根据方法组合得分
    if method == 'correlation':
        final_scores = feature_scores['correlation']
    elif method == 'mutual_info':
        final_scores = feature_scores['mutual_info']
    else:  # both
        # 归一化两个得分并取平均
        corr_scores = pd.Series(feature_scores['correlation'])
        mi_scores = pd.Series(feature_scores['mutual_info'])
        
        # 归一化到[0,1]区间
        corr_scores_norm = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min())
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        
        # 加权平均 (可以调整权重)
        final_scores = (0.5 * corr_scores_norm + 0.5 * mi_scores_norm).to_dict()
    
    # 根据得分排序并选择top特征
    sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    n_features_to_keep = int(len(feature_cols) * keep_ratio)
    selected_features = [feat[0] for feat in sorted_features[:n_features_to_keep]]
    
    # 总是保留目标列
    selected_features.append(target_col)
    
    print(f"特征选择完成:")
    print(f"  - 原始特征数: {len(feature_cols)}")
    print(f"  - 保留特征数: {len(selected_features)-1}")  # 减1是因为不计算目标列
    print(f"  - 删除特征数: {len(feature_cols) - (len(selected_features)-1)}")
    
    # Top 10 特征得分
    print("Top 10 特征得分:")
    for i, (feat, score) in enumerate(sorted_features[:10]):
        print(f"  {i+1:2d}. {feat}: {score:.4f}")
    
    df_selected = df[selected_features]
    return df_selected, selected_features, feature_scores

def handle_outliers(df, method='clip', std_threshold=3, target_col='label'):
    """
    处理异常值
    
    Args:
        df: DataFrame
        method: 'clip' 截断, 'remove' 删除, 'winsorize' Winsorize
        std_threshold: 标准差阈值
        target_col: 目标列名，不处理异常值
    
    Returns:
        df_processed: 处理后的DataFrame
        outlier_info: 异常值处理信息
    """
    print(f"=== 处理异常值 (方法: {method}) ===")
    
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        feature_cols = [col for col in numeric_cols if col != target_col]
    else:
        feature_cols = numeric_cols
    
    outlier_info = {}
    
    for col in feature_cols:
        if df_processed[col].std() == 0:
            continue  # 跳过常数列
            
        mean_val = df_processed[col].mean()
        std_val = df_processed[col].std()
        
        lower_bound = mean_val - std_threshold * std_val
        upper_bound = mean_val + std_threshold * std_val
        
        outliers = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            outlier_info[col] = outlier_count
            
            if method == 'clip':
                # 截断到边界值
                df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            elif method == 'remove':
                # 删除包含异常值的行（谨慎使用）
                df_processed = df_processed[~outliers]
    
    if outlier_info:
        total_outliers = sum(outlier_info.values())
        print(f"处理了 {len(outlier_info)} 个特征的异常值，总计 {total_outliers} 个")
    else:
        print("未发现需要处理的异常值")

    return df_processed, outlier_info

def process_data(df, config=None, is_training_data=True):
    """
    主数据处理函数，整合所有预处理步骤
    
    Args:
        df: 原始DataFrame
        config: 配置字典
        is_training_data: 是否为训练数据
    
    Returns:
        df_processed: 处理后的DataFrame
    """
    print(f"\n{'='*50}")
    print(f"开始数据预处理 ({'训练数据' if is_training_data else '测试数据'})")
    print(f"原始数据形状: {df.shape}")
    print(f"{'='*50}")
    
    df_processed = df.copy()
    
    processing_config = config["data_processing"]
    
    target_col = config.get("data", {}).get("target_column", "label") if config else "label"
    
    # 1. 处理缺失值和无穷值
    df_processed, removed_missing = handle_missing_values(
        df_processed, 
        missing_threshold=processing_config.get("missing_threshold", 0.2),
        method=processing_config.get("missing_fill_method", "ffill")
    )
    
    # 2. 删除低方差特征
    df_processed, removed_low_var = remove_low_variance_features(
        df_processed,
        variance_threshold=processing_config.get("variance_threshold", 1e-6),
        target_col=target_col
    )
    
    # # 3. 删除高度相关特征
    # df_processed, removed_corr = remove_highly_correlated_features(
    #     df_processed,
    #     correlation_threshold=processing_config.get("correlation_threshold", 0.95),
    #     target_col=target_col
    # )
    
    # 4. 基于目标变量的特征选择（仅对训练数据）
    if is_training_data and target_col in df_processed.columns:
        df_processed, selected_features, feature_scores = select_features_by_target_correlation(
            df_processed,
            target_col=target_col,
            keep_ratio=processing_config.get("feature_keep_ratio", 0.7),
            method=processing_config.get("feature_selection_method", "both")
        )
    
    # 5. 处理异常值
    df_processed, outlier_info = handle_outliers(
        df_processed,
        method=processing_config.get("outlier_method", "clip"),
        std_threshold=processing_config.get("outlier_std_threshold", 3),
        target_col=target_col
    )
    
    print(f"\n{'='*50}")
    print(f"数据预处理完成!")
    print(f"最终数据形状: {df_processed.shape}")
    print(f"特征数变化: {df.shape[1]} -> {df_processed.shape[1]} ({df_processed.shape[1] - df.shape[1]:+d})")
    print(f"{'='*50}\n")
    
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