import numpy as np
import pandas as pd
import os
from alpha_vantage.timeseries import TimeSeries
from torch.utils.data import Dataset
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler, QuantileTransformer
import warnings
warnings.filterwarnings('ignore')

def load_parquet_data(config):
    """
    加载本地parquet数据文件，进行数据清洗和预处理(现在只用在无特征工程版本了)
    """
    print("Loading data from parquet files...")
    train_df = pd.read_parquet(config["data"]["train_path"])
    test_df = pd.read_parquet(config["data"]["test_path"])

    # 数据采前sample_size个样本
    train_df_sample = train_df.head(config["data"]["sample_size"])
    test_df_sample = test_df.head(config["data"]["test_sample_size"])

    # 先处理训练数据，获取预处理策略
    train_cleaned, preprocessing_info = process_data(train_df_sample, config, is_training_data=True)

    # 使用训练数据的预处理策略处理测试数据
    test_cleaned = process_data(test_df_sample, config, is_training_data=False, preprocessing_info=preprocessing_info)
    
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
    
    protected_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"]
    
    # 首先将无穷值转换为NaN，统一处理
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df_processed[numeric_cols]).sum().sum()
    if inf_count > 0:
        print(f"发现 {inf_count} 个无穷值，将其转换为NaN统一处理")
        df_processed[numeric_cols] = df_processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 计算每列的缺失比例（包括原来的NaN和转换后的无穷值）
    missing_ratios = df_processed.isnull().sum() / len(df_processed)
    
    # 识别需要删除的特征（缺失比例过高）
    features_to_remove_candidates = missing_ratios[missing_ratios > missing_threshold].index.tolist()
    features_to_remove = [f for f in features_to_remove_candidates if f not in protected_features]
    
    if features_to_remove:
        print(f"删除缺失比例>{missing_threshold:.1%}的特征: {len(features_to_remove)}个")
        df_processed = df_processed.drop(columns=features_to_remove)
    
    # 处理剩余的缺失值（包括NaN和无穷值）
    remaining_missing = df_processed.isnull().sum().sum()
    if remaining_missing > 0:
        if method == 'ffill':
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        elif method == 'bfill':
            df_processed = df_processed.fillna(method='bfill').fillna(method='ffill')
        elif method == 'interpolate':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].interpolate(method='linear')
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
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
    
    protected_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        feature_cols = [col for col in numeric_cols if col != target_col]
    else:
        feature_cols = numeric_cols
    
    # 计算方差，找出低方差特征
    variances = df[feature_cols].var()
    
    low_variance_features_candidates = variances[variances <= variance_threshold].index.tolist()
    low_variance_features = [f for f in low_variance_features_candidates if f not in protected_features]

    if low_variance_features:
        # print(f"删除方差<={variance_threshold}的特征: {len(low_variance_features)}个")
        df_filtered = df.drop(columns=low_variance_features)
    else:
        # print("未发现需要删除的低方差特征")
        df_filtered = df.copy()
        
    print(f"方差过滤完成，保留特征数: {df_filtered.shape[1]}")
    return df_filtered, low_variance_features

def remove_highly_correlated_features(df, correlation_threshold=0.95, target_col='label', sample_size=10000):
    """
    删除高度相关的特征
    
    Args:
        df: DataFrame
        correlation_threshold: 相关系数阈值
        target_col: 目标列名，不参与相关性计算
        sample_size: 计算相关性矩阵时的采样数量
    
    Returns:
        df_filtered: 过滤后的DataFrame
        removed_features: 被删除的特征列表
    """
    print("=== 删除高度相关特征 ===")
    protected_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"]
    
    # 获取特征列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        feature_cols = [col for col in numeric_cols if col != target_col]
    else:
        feature_cols = numeric_cols
    
    print("计算特征相关矩阵...")
    actual_sample_size = min(sample_size, len(df))# 使用少量数据计算相关矩阵
    if len(df) > actual_sample_size:
        sample_idx = np.random.choice(len(df), actual_sample_size, replace=False)
        df_sample = df.iloc[sample_idx]
        corr_matrix = df_sample[feature_cols].corr().abs()
    else:
        corr_matrix = df[feature_cols].corr().abs()
    
    # 找出高度相关的特征对
    removed_features_candidates = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > correlation_threshold:
                # 删除第二个特征
                feature_to_remove = corr_matrix.columns[j]
                if feature_to_remove not in removed_features_candidates:
                    removed_features_candidates.append(feature_to_remove)
    
    removed_features = [f for f in removed_features_candidates if f not in protected_features]
    
    if removed_features:
        # print(f"删除相关系数>{correlation_threshold}的特征: {len(removed_features)}个")
        df_filtered = df.drop(columns=removed_features)
    else:
        # print("未发现需要删除的高度相关特征")
        df_filtered = df.copy()
    
    print(f"相关性过滤完成，保留特征数: {df_filtered.shape[1]}")
    return df_filtered, removed_features

def select_features_by_target_correlation(df, target_col='label', keep_ratio=0.7, method='both', mutual_info_sample_size=20000):
    """
    基于与目标变量的相关性选择特征
    
    Args:
        df: DataFrame
        target_col: 目标列名
        keep_ratio: 保留特征的比例
        method: 'correlation' 线性相关, 'mutual_info' 互信息, 'both' 两者结合
        mutual_info_sample_size: 计算互信息时的采样数量
    
    Returns:
        df_selected: 特征选择后的DataFrame
        selected_features: 保留的特征列表
        feature_scores: 特征评分字典
    """
    print(f"=== 基于与目标变量相关性的特征选择 (保留{keep_ratio:.1%}) ===")
    protected_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"]

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
    
    # 计算互信息
    if method in ['mutual_info', 'both']:
        print("计算与目标变量的互信息...")

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # 使用少量数据计算互信息
        actual_sample_size = min(mutual_info_sample_size, len(X_scaled))
        if len(X_scaled) > actual_sample_size:
            sample_idx = np.random.choice(len(X_scaled), actual_sample_size, replace=False)
            X_sample = X_scaled.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X_scaled
            y_sample = y

        mutual_info_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
        feature_scores['mutual_info'] = dict(zip(feature_cols, mutual_info_scores))
    
    # 组合相关性分数
    if method == 'correlation':
        final_scores = feature_scores['correlation']
    elif method == 'mutual_info':
        final_scores = feature_scores['mutual_info']
    else:  # both
        corr_scores = pd.Series(feature_scores['correlation'])
        mi_scores = pd.Series(feature_scores['mutual_info'])
        
        corr_scores_norm = (corr_scores - corr_scores.min()) / (corr_scores.max() - corr_scores.min()) # 归一化到[0,1]区间
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        
        final_scores = (0.5 * corr_scores_norm + 0.5 * mi_scores_norm).to_dict() # 加权平均
    
    # 根据得分排序并选择top特征
    sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    n_features_to_keep = int(len(feature_cols) * keep_ratio)
    selected_features = [feat[0] for feat in sorted_features[:n_features_to_keep]]

    # 检查并补充保护列
    for pf in protected_features:
        if pf not in selected_features:
            selected_features.append(pf)
    
    print(f"特征选择完成: 保留特征数: {len(selected_features)-1}")

    df_selected = df[selected_features]
    return df_selected, selected_features, feature_scores

def handle_financial_outliers(df, method='iqr', window='1D', target_col='label'):
    """
    针对金融时间序列数据的异常值处理
    
    Args:
        df: DataFrame
        method: 'iqr' (基于四分位距), 'quantile' (基于分位数), 'zscore' (基于z-score)
        window: 时间窗口大小，用于动态计算阈值，如'1D'表示按天
        target_col: 目标变量列名
    
    Returns:
        df_processed: 处理后的DataFrame
        outlier_info: 异常值处理信息
    """
    print(f"=== 异常值处理 (方法: {method}, 窗口: {window}) ===")
    
    df_processed = df.copy()

    protected_cols = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"]
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in protected_cols]
    
    outlier_info = {}
    
    # 按时间窗口分组处理
    for time_window in df_processed.groupby(pd.Grouper(freq=window)):
        window_data = time_window[1]
        
        for col in feature_cols:
            if window_data[col].std() == 0:
                continue  # 跳过常数列
                
            if method == 'iqr':
                # IQR方法
                Q1 = window_data[col].quantile(0.25)
                Q3 = window_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            elif method == 'quantile':
                # 分位数方法
                lower_bound = window_data[col].quantile(0.01)
                upper_bound = window_data[col].quantile(0.99)
            
            elif method == 'zscore':
                # 改进的Z-score方法，使用中位数和MAD
                median = window_data[col].median()
                mad = np.median(np.abs(window_data[col] - median))
                modified_zscore = 0.6745 * (window_data[col] - median) / mad
                outliers = np.abs(modified_zscore) > 3.5
                
                if outliers.sum() > 0:
                    # 使用EWMA进行平滑处理
                    df_processed.loc[window_data.index[outliers], col] = \
                        df_processed.loc[window_data.index[outliers], col].ewm(span=5).mean()
                continue
            
            # 对于iqr和quantile方法的处理
            if method in ['iqr', 'quantile']:
                outliers = (window_data[col] < lower_bound) | (window_data[col] > upper_bound)
                if outliers.sum() > 0:
                    # 记录异常值信息
                    if col not in outlier_info:
                        outlier_info[col] = 0
                    outlier_info[col] += outliers.sum()
                    
                    # 使用EWMA平滑处理异常值
                    df_processed.loc[window_data.index[outliers], col] = \
                        df_processed.loc[window_data.index[outliers], col].ewm(span=5).mean()
    
    if outlier_info:
        total_outliers = sum(outlier_info.values())
        print(f"处理了 {len(outlier_info)} 个特征的异常值，总计 {total_outliers} 个")
    else:
        print("未发现需要处理的异常值")
        
    return df_processed, outlier_info

def process_data(df, config=None, is_training_data=True, preprocessing_info=None):
    """
    数据清洗和基础预处理函数
    
    Args:
        df: 输入数据DataFrame
        config: 配置字典
        is_training_data: 是否为训练数据，影响预处理策略的计算
        preprocessing_info: 预处理信息（用于测试数据保持一致性）
    
    Returns:
        df_processed: 处理后的DataFrame
        preprocessing_info: 预处理信息字典（仅训练数据返回）
    """
    print(f"开始{'训练' if is_training_data else '测试'}数据预处理...")
    
    df_processed = df.copy()
    
    # 从配置中获取预处理参数
    processing_config = config.get("data_processing", {}) if config else {}
    target_col = config.get("data", {}).get("target_column", "label") if config else "label"
    
    # 初始化预处理信息记录
    current_preprocessing_info = {}
    
    # 处理缺失值和无穷值
    df_processed, removed_missing = handle_missing_values(
        df_processed,
        missing_threshold=processing_config.get("missing_threshold", 0.2),
        method=processing_config.get("missing_method", "ffill")
    )
    
    # 删除低方差特征
    if is_training_data:
        df_processed, removed_low_var = remove_low_variance_features(
            df_processed,
            variance_threshold=processing_config.get("variance_threshold", 1e-6),
            target_col=target_col
        )
        if is_training_data:
            current_preprocessing_info['removed_low_var'] = removed_low_var
    else:
        # 测试数据：删除训练时确定的低方差特征
        removed_low_var = preprocessing_info['removed_low_var']
        if removed_low_var:
            existing_low_var = [col for col in removed_low_var if col in df_processed.columns]
            if existing_low_var:
                print(f"=== 应用训练数据的低方差过滤策略 ===")
                print(f"删除特征: {len(existing_low_var)}个")
                df_processed = df_processed.drop(columns=existing_low_var)
    
    # 删除高度相关特征
    if is_training_data:
        df_processed, removed_corr = remove_highly_correlated_features(
            df_processed,
            correlation_threshold=processing_config.get("correlation_threshold", 0.95),
            target_col=target_col,
            sample_size=processing_config.get("correlation_sample_size", 10000)
        )
        if is_training_data:
            current_preprocessing_info['removed_corr'] = removed_corr
    else:
        # 测试数据：删除训练时确定的高相关特征
        removed_corr = preprocessing_info['removed_corr']
        if removed_corr:
            existing_corr = [col for col in removed_corr if col in df_processed.columns]
            if existing_corr:
                print(f"=== 应用训练数据的相关性过滤策略 ===")
                print(f"删除特征: {len(existing_corr)}个")
                df_processed = df_processed.drop(columns=existing_corr)
    
    # 基于与目标变量相关性的特征选择
    if is_training_data and target_col in df_processed.columns:
        df_processed, selected_features, feature_scores = select_features_by_target_correlation(
            df_processed,
            target_col=target_col,
            keep_ratio=processing_config.get("feature_keep_ratio", 0.7),
            method=processing_config.get("feature_selection_method", "both"),
            mutual_info_sample_size=processing_config.get("mutual_info_sample_size", 20000)
        )
        current_preprocessing_info['selected_features'] = selected_features
        current_preprocessing_info['feature_scores'] = feature_scores
    else:
        # 测试数据：使用训练数据确定的特征列表
        selected_features = preprocessing_info['selected_features']
        print(f"=== 应用训练数据的特征选择策略 ===")
        print(f"保留特征: {len(selected_features)}个")
        
        # 只保留训练数据中选择的特征
        available_features = [col for col in selected_features if col in df_processed.columns]
        missing_features = [col for col in selected_features if col not in df_processed.columns]
        
        if missing_features:
            print(f"警告: 测试数据中缺少 {len(missing_features)} 个特征: {missing_features[:10]}...")
        
        if available_features:
            df_processed = df_processed[available_features]
        else:
            raise ValueError("测试数据中没有任何训练数据中选择的特征！")
    
    # 处理异常值
    df_processed, outlier_info = handle_financial_outliers(
        df_processed,
        method=processing_config.get("outlier_method", "iqr"),
        window=processing_config.get("outlier_window", "1D"),
        target_col=target_col
    )
    
    if is_training_data:
        current_preprocessing_info['outlier_info'] = outlier_info

    print(f"\n{'='*50}")
    print(f"基础数据预处理完成!")
    print(f"最终数据形状: {df_processed.shape}")
    print(f"特征数变化: {df.shape[1]} -> {df_processed.shape[1]} ({df_processed.shape[1] - df.shape[1]:+d})")
    print(f"{'='*50}\n")
    
    if is_training_data:
        return df_processed, current_preprocessing_info
    else:
        return df_processed

def load_processed_data(config, use_cache=True):
    """
    加载清理后的parquet数据，用于在【无特征工程版本】preprocess中保存为pickle
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