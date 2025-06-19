"""
数据预处理模块
负责加载原始parquet数据，进行预处理，并保存为可直接用于训练的格式
python -m dataset.preprocess
"""

import numpy as np
import pandas as pd
import os
import pickle
from typing import Tuple, List
from .data_loader import load_processed_data, prepare_data_x, prepare_data_y
from utils.normalizer import Normalizer
from config import CONFIG
import argparse

def create_data_directory(data_dir: str = "data") -> str:
    """创建数据目录"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"创建数据目录: {data_dir}")
    return data_dir

def select_features(train_df: pd.DataFrame, config: dict) -> List[str]:
    """
    特征选择
    """
    # 确定特征列
    all_feature_cols = [col for col in train_df.columns if col != config["data"]["target_column"]]
    target_col = config["data"]["target_column"]
    
    # 特征选择
    if config["data"]["feature_selection"] == "all":
        feature_cols = all_feature_cols
        print(f"使用所有 {len(feature_cols)} 个特征")
    elif config["data"]["feature_selection"] == "top_k":
        # 基于与目标变量的相关性选择top_k特征
        print("正在计算特征相关性...")
        correlations = train_df[all_feature_cols].corrwith(train_df[target_col]).abs()
        top_features = correlations.nlargest(config["data"]["top_k_features"]).index.tolist()
        feature_cols = top_features
        print(f"选择了前 {len(feature_cols)} 个最相关的特征")
        print(f"选择的特征: {feature_cols}")
    else:
        # 自定义特征列表
        feature_cols = config["data"]["feature_selection"]
        print(f"使用自定义特征: {len(feature_cols)} 个")
    
    return feature_cols

def preprocess_data(sample_size: int = 1000, 
                   feature_count: int = 10, 
                   window_size: int = 10,
                   test_sample_size: int = 200,
                   save_dir: str = "data") -> dict:
    """
    数据预处理主函数
    """
    config = CONFIG.copy()
    config["data"]["feature_selection"] = "all" # top_k
    config["data"]["top_k_features"] = feature_count
    config["data"]["window_size"] = window_size

    # 创建保存目录
    data_dir = create_data_directory(save_dir)
    
    # 加载清理过后的parquet数据
    train_df, test_df = load_processed_data(config, use_cache=True)

    # 数据采前sample_size个样本
    train_df_sample = train_df.head(sample_size)
    test_df_sample = test_df.head(test_sample_size)

    # 特征选择
    feature_cols = select_features(train_df_sample, config)
    target_col = config["data"]["target_column"]
    
    # 准备时间序列窗口数据
    data_x_train, _ = prepare_data_x(train_df_sample, window_size, feature_cols)
    data_y_train = prepare_data_y(train_df_sample, window_size, target_col)

    data_x_test, _ = prepare_data_x(test_df_sample, window_size, feature_cols)
    data_y_test = prepare_data_y(test_df_sample, window_size, target_col)
    
    print(f"训练数据形状: {data_x_train.shape}")  # (n_samples, window_size, n_features)
    print(f"训练目标数据形状: {data_y_train.shape}")  # (n_samples,)
    print(f"测试数据形状: {data_x_test.shape}")  # (n_samples, window_size, n_features)
    print(f"测试目标数据形状: {data_y_test.shape}")  # (n_samples,)
    
    # 数据标准化
    feature_scaler = Normalizer()
    target_scaler = Normalizer()

    original_shape_train = data_x_train.shape
    original_shape_test = data_x_test.shape
    data_x_reshaped_train = data_x_train.reshape(-1, data_x_train.shape[-1])  # (n_samples * window_size, n_features)
    data_x_reshaped_test = data_x_test.reshape(-1, data_x_test.shape[-1])  # (n_samples * window_size, n_features)
    data_x_normalized_reshaped_train = feature_scaler.fit_transform(data_x_reshaped_train)
    data_x_normalized_reshaped_test = feature_scaler.fit_transform(data_x_reshaped_test)
    data_x_normalized_train = data_x_normalized_reshaped_train.reshape(original_shape_train)
    data_x_normalized_test = data_x_normalized_reshaped_test.reshape(original_shape_test)

    data_y_normalized_train = target_scaler.fit_transform(data_y_train.reshape(-1, 1)).flatten()
    data_y_normalized_test = target_scaler.fit_transform(data_y_test.reshape(-1, 1)).flatten()

    # 分割数据集
    split_ratio = config["data"]["train_split_size"]
    split_index = int(data_y_normalized_train.shape[0] * split_ratio)
    
    data_x_train = data_x_normalized_train[:split_index]
    data_x_val = data_x_normalized_train[split_index:]
    data_y_train = data_y_normalized_train[:split_index]
    data_y_val = data_y_normalized_train[split_index:]

    data_x_test = data_x_normalized_test
    data_y_test = data_y_normalized_test

    train_data = {
        'x': data_x_train,
        'y': data_y_train
    }
    
    val_data = {
        'x': data_x_val,
        'y': data_y_val
    }
    
    test_data = {
        'x': data_x_test,
        'y': data_y_test
    }
    
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    
    metadata = {
        'feature_cols': feature_cols,
        'target_col': target_col,
        'window_size': window_size,
        'sample_size': sample_size,
        'feature_count': len(feature_cols),
        'original_data_shape': train_df_sample.shape,
        'processed_data_shapes': {
            'train_x': data_x_train.shape,
            'train_y': data_y_train.shape,
            'val_x': data_x_val.shape,
            'val_y': data_y_val.shape,
            'test_x': data_x_test.shape,
            'test_y': data_y_test.shape
        },
        'split_ratio': split_ratio,
        'config': config
    }
    
    files_saved = []
    with open(os.path.join(data_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
        files_saved.append('train_data.pkl')
    with open(os.path.join(data_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
        files_saved.append('val_data.pkl')
    with open(os.path.join(data_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
        files_saved.append('test_data.pkl')
    with open(os.path.join(data_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
        files_saved.append('scalers.pkl')
    with open(os.path.join(data_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
        files_saved.append('metadata.pkl')
    
    summary = {
        'success': True,
        'save_dir': data_dir,
        'files_saved': files_saved,
        'metadata': metadata,
        'summary_stats': {
            'original_samples': len(train_df_sample) + len(test_df_sample),
            'train_samples': len(data_y_train),
            'val_samples': len(data_y_val),
            'test_samples': len(data_y_test),
            'features_selected': len(feature_cols),
            'window_size': window_size
        }
    }
    return summary

def check_preprocessed_data_exists(data_dir: str = "data") -> bool:
    """
    检查预处理数据是否存在
    """
    required_files = [
        'train_data.pkl',
        'val_data.pkl', 
        'test_data.pkl',
        'scalers.pkl',
        'metadata.pkl'
    ]
    
    if not os.path.exists(data_dir):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            return False
    
    return True

def load_preprocessed_data(data_dir: str = "data") -> dict:
    """
    在主程序中加载预处理后的pickle数据
    """
    if not check_preprocessed_data_exists(data_dir):
        raise FileNotFoundError(f"请先运行数据预处理。")
    
    with open(os.path.join(data_dir, 'train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, 'val_data.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(data_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    with open(os.path.join(data_dir, 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'scalers': scalers,
        'metadata': metadata
    }

if __name__ == "__main__":
    config0 = CONFIG.copy()
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--sample_size', type=int, default=config0["data"]["sample_size"], help='使用的样本数量')
    parser.add_argument('--feature_count', type=int, default=30, help='使用的特征数量')
    parser.add_argument('--window_size', type=int, default=60, help='时间窗口大小')
    parser.add_argument('--test_sample_size', type=int, default=config0["data"]["test_sample_size"], help='测试样本数量')
    parser.add_argument('--save_dir', type=str, default='data', help='保存目录')
    
    args = parser.parse_args()
    
    summary = preprocess_data(
        sample_size=args.sample_size,
        feature_count=args.feature_count,
        window_size=args.window_size,
        test_sample_size=args.test_sample_size,
        save_dir=args.save_dir
    )

    print("数据预处理成功完成!")
