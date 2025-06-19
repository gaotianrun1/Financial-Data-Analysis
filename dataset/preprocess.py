"""
数据预处理模块
负责加载原始parquet数据，进行预处理和特征工程，并保存为可直接用于训练的格式
python -m dataset.preprocess --enable_feature_engineering
"""

import numpy as np
import pandas as pd
import os
import pickle
from typing import Tuple, List
from .data_loader import load_processed_data, prepare_data_x, prepare_data_y
from .feature_engineering import apply_integrated_feature_engineering
from utils.normalizer import Normalizer
from config import CONFIG
import argparse

def create_data_directory(data_dir: str = "data") -> str:
    """创建数据目录"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"创建数据目录: {data_dir}")
    return data_dir



def preprocess_data_with_feature_engineering(sample_size: int = 1000, 
                                            window_size: int = 10,
                                            test_sample_size: int = 200,
                                            save_dir: str = "data",
                                            enable_feature_engineering: bool = True) -> dict:
    """
    集成特征工程的数据预处理主函数
    
    实现三步数据处理流程：
    1. 初步数据清理（data_loader.process_data）
    2. 特征工程（feature_engineering）
    3. 再次数据清理（data_loader中的清洗函数）
    """
    config = CONFIG.copy()
    
    # 启用特征工程配置
    config["feature_engineering"]["enable_feature_engineering"] = enable_feature_engineering
    config["data"]["window_size"] = window_size

    # 创建保存目录
    data_dir = create_data_directory(save_dir)
    
    # 第一步：加载原始数据并进行初步清理
    print("="*60)
    print("第一步：加载原始数据并进行初步清理")
    print("="*60)
    
    # 加载原始parquet数据
    print("加载原始parquet数据...")
    train_df = pd.read_parquet(config["data"]["train_path"])
    test_df = pd.read_parquet(config["data"]["test_path"])
    
    # 采样
    train_df_sample = train_df.head(sample_size)
    test_df_sample = test_df.head(test_sample_size)

    from .data_loader import process_data
    # 训练数据初步清理
    train_df_basic, preprocessing_info = process_data(
        train_df_sample, 
        config, 
        is_training_data=True
    )
    # 测试数据初步清理
    test_df_basic = process_data(
        test_df_sample, 
        config, 
        is_training_data=False, 
        preprocessing_info=preprocessing_info
    )
    
    target_col = config["data"]["target_column"]
    
    # 第二步：特征工程
    print("\n" + "="*60)
    print("第二步：特征工程")
    print("="*60)
    
    if enable_feature_engineering:
        print("开始应用特征工程...")
        print(f"训练数据初步清理后形状: {train_df_basic.shape}")
        print(f"测试数据初步清理后形状: {test_df_basic.shape}")
        
        # 对训练数据应用特征工程
        train_df_enhanced, feature_engineering_info = apply_integrated_feature_engineering(
            train_df_basic, config, target_col
        )
        
        # 对测试数据应用特征工程
        test_df_enhanced, _ = apply_integrated_feature_engineering(
            test_df_basic, config, target_col
        )
        
        print(f"训练数据特征工程后形状: {train_df_enhanced.shape}")
        print(f"测试数据特征工程后形状: {test_df_enhanced.shape}")
        
        # 确保两个数据集有相同的特征
        train_features = set(train_df_enhanced.columns)
        test_features = set(test_df_enhanced.columns)
        
        common_features = list(train_features.intersection(test_features))
        if target_col in common_features:
            feature_cols = [col for col in common_features if col != target_col]
        else:
            raise ValueError(f"目标列 {target_col} 在处理后的数据中不存在！")
        
        print(f"共同特征数量: {len(feature_cols)}")
        
        # 保持特征对齐
        train_df_processed = train_df_enhanced[feature_cols + [target_col]]
        test_df_processed = test_df_enhanced[feature_cols + [target_col]]
        
    else:
        print("跳过特征工程，使用初步清理后的数据...")
        train_df_processed = train_df_basic
        test_df_processed = test_df_basic
        feature_cols = [col for col in train_df_basic.columns if col != target_col]
        feature_engineering_info = {}
    
    print("\n" + "="*60)
    print("第三步：时间序列窗口化和标准化")
    print("="*60)
    
    print(f"最终训练数据形状: {train_df_processed.shape}")
    print(f"最终测试数据形状: {test_df_processed.shape}")
    print(f"特征数量: {len(feature_cols)}")
    
    # 准备时间序列窗口数据
    data_x_train, _ = prepare_data_x(train_df_processed, window_size, feature_cols)
    data_y_train = prepare_data_y(train_df_processed, window_size, target_col)

    data_x_test, _ = prepare_data_x(test_df_processed, window_size, feature_cols)
    data_y_test = prepare_data_y(test_df_processed, window_size, target_col)
    
    print(f"时间序列数据形状:")
    print(f"  训练X: {data_x_train.shape}, 训练Y: {data_y_train.shape}")
    print(f"  测试X: {data_x_test.shape}, 测试Y: {data_y_test.shape}")
    
    # 第四步：数据标准化
    print(f"\n第四步：数据标准化...")
    feature_scaler = Normalizer()
    target_scaler = Normalizer()

    # 特征标准化
    original_shape_train = data_x_train.shape
    original_shape_test = data_x_test.shape
    
    data_x_reshaped_train = data_x_train.reshape(-1, data_x_train.shape[-1])
    data_x_reshaped_test = data_x_test.reshape(-1, data_x_test.shape[-1])
    
    data_x_normalized_reshaped_train = feature_scaler.fit_transform(data_x_reshaped_train)
    data_x_normalized_reshaped_test = feature_scaler.transform(data_x_reshaped_test)
    
    data_x_normalized_train = data_x_normalized_reshaped_train.reshape(original_shape_train)
    data_x_normalized_test = data_x_normalized_reshaped_test.reshape(original_shape_test)

    # 目标变量标准化
    data_y_normalized_train = target_scaler.fit_transform(data_y_train.reshape(-1, 1)).flatten()
    data_y_normalized_test = target_scaler.transform(data_y_test.reshape(-1, 1)).flatten()

    # 第五步：分割数据集和保存
    print(f"\n第五步：数据分割和保存...")
    split_ratio = config["data"]["train_split_size"]
    split_index = int(data_y_normalized_train.shape[0] * split_ratio)
    
    data_x_train_final = data_x_normalized_train[:split_index]
    data_x_val = data_x_normalized_train[split_index:]
    data_y_train_final = data_y_normalized_train[:split_index]
    data_y_val = data_y_normalized_train[split_index:]

    # 构建返回数据
    train_data = {'x': data_x_train_final, 'y': data_y_train_final}
    val_data = {'x': data_x_val, 'y': data_y_val}
    test_data = {'x': data_x_normalized_test, 'y': data_y_normalized_test}
    
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
        'feature_engineering_enabled': enable_feature_engineering,
        'original_data_shape': train_df_sample.shape,
        'basic_cleaned_shape': train_df_basic.shape,
        'final_processed_shape': train_df_processed.shape,
        'processed_data_shapes': {
            'train_x': data_x_train_final.shape,
            'train_y': data_y_train_final.shape,
            'val_x': data_x_val.shape,
            'val_y': data_y_val.shape,
            'test_x': data_x_normalized_test.shape,
            'test_y': data_y_normalized_test.shape
        },
        'split_ratio': split_ratio,
        'config': config,
        'preprocessing_info': preprocessing_info
    }
    
    if enable_feature_engineering:
        metadata['feature_engineering_info'] = feature_engineering_info
    
    # 保存数据
    files_saved = []
    save_files = {
        'train_data.pkl': train_data,
        'val_data.pkl': val_data,
        'test_data.pkl': test_data,
        'scalers.pkl': scalers,
        'metadata.pkl': metadata
    }
    
    for filename, data in save_files.items():
        with open(os.path.join(data_dir, filename), 'wb') as f:
            pickle.dump(data, f)
        files_saved.append(filename)
    
    summary = {
        'success': True,
        'save_dir': data_dir,
        'files_saved': files_saved,
        'metadata': metadata,
        'summary_stats': {
            'original_samples': len(train_df_sample) + len(test_df_sample),
            'train_samples': len(data_y_train_final),
            'val_samples': len(data_y_val),
            'test_samples': len(data_y_normalized_test),
            'features_selected': len(feature_cols),
            'window_size': window_size,
            'feature_engineering_enabled': enable_feature_engineering
        }
    }
    
    print(f"\n{'='*60}")
    print(f"数据预处理完成!")
    print(f"保存目录: {data_dir}")
    print(f"特征工程: {'启用' if enable_feature_engineering else '禁用'}")
    print(f"最终特征数: {len(feature_cols)}")
    
    return summary

def preprocess_data(sample_size: int = 1000, 
                   window_size: int = 10,
                   test_sample_size: int = 200,
                   save_dir: str = "data") -> dict:
    """
    传统数据预处理主函数
    """
    config = CONFIG.copy()
    config["data"]["window_size"] = window_size

    # 创建保存目录
    data_dir = create_data_directory(save_dir)
    
    # 加载清理过后的parquet数据
    train_df, test_df = load_processed_data(config, use_cache=True)

    # 数据采前sample_size个样本
    train_df_sample = train_df.head(sample_size)
    test_df_sample = test_df.head(test_sample_size)

    # 获取特征列（应该来自预处理后的数据）
    target_col = config["data"]["target_column"]
    
    # 确保训练和测试数据具有相同的特征列（除了目标列）
    train_feature_cols = [col for col in train_df_sample.columns if col != target_col]
    test_feature_cols = [col for col in test_df_sample.columns if col != target_col]
    
    # 取交集确保特征一致性
    common_feature_cols = list(set(train_feature_cols) & set(test_feature_cols))
    
    print(f"训练数据特征数: {len(train_feature_cols)}")
    print(f"测试数据特征数: {len(test_feature_cols)}")
    print(f"共同特征数: {len(common_feature_cols)}")
    
    if len(common_feature_cols) == 0:
        raise ValueError("训练数据和测试数据没有共同的特征！")
    
    feature_cols = common_feature_cols
        
    # # 确保测试数据也只包含选择的特征
    # train_df_sample = train_df_sample[feature_cols + [target_col]]
    # test_df_sample = test_df_sample[feature_cols + [target_col]]
    
    # 准备时间序列窗口数据
    data_x_train, _ = prepare_data_x(train_df_sample, window_size, feature_cols)
    data_y_train = prepare_data_y(train_df_sample, window_size, target_col)

    data_x_test, _ = prepare_data_x(test_df_sample, window_size, feature_cols)
    data_y_test = prepare_data_y(test_df_sample, window_size, target_col)
    
    print(f"训练数据形状: {data_x_train.shape}")  # (n_samples, window_size, n_features)
    print(f"训练目标数据形状: {data_y_train.shape}")  # (n_samples,)
    print(f"测试数据形状: {data_x_test.shape}")  # (n_samples, window_size, n_features)
    print(f"测试目标数据形状: {data_y_test.shape}")  # (n_samples,)
    
    # 数据标准化 - 只在训练数据上fit，然后transform测试数据
    feature_scaler = Normalizer()
    target_scaler = Normalizer()

    original_shape_train = data_x_train.shape
    original_shape_test = data_x_test.shape
    
    # 特征标准化：在训练数据上fit，然后分别transform训练和测试数据
    data_x_reshaped_train = data_x_train.reshape(-1, data_x_train.shape[-1])  # (n_samples * window_size, n_features)
    data_x_reshaped_test = data_x_test.reshape(-1, data_x_test.shape[-1])  # (n_samples * window_size, n_features)
    
    # 在训练数据上fit并transform
    data_x_normalized_reshaped_train = feature_scaler.fit_transform(data_x_reshaped_train)
    # 使用训练数据的统计信息transform测试数据
    data_x_normalized_reshaped_test = feature_scaler.transform(data_x_reshaped_test)
    
    data_x_normalized_train = data_x_normalized_reshaped_train.reshape(original_shape_train)
    data_x_normalized_test = data_x_normalized_reshaped_test.reshape(original_shape_test)

    # 目标变量标准化：在训练数据上fit，然后分别transform训练和测试数据
    data_y_normalized_train = target_scaler.fit_transform(data_y_train.reshape(-1, 1)).flatten()
    data_y_normalized_test = target_scaler.transform(data_y_test.reshape(-1, 1)).flatten()

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
    parser.add_argument('--window_size', type=int, default=60, help='时间窗口大小')
    parser.add_argument('--test_sample_size', type=int, default=config0["data"]["test_sample_size"], help='测试样本数量')
    parser.add_argument('--save_dir', type=str, default='data', help='保存目录')
    parser.add_argument('--use_traditional', action='store_true', help='使用传统预处理方法')
    parser.add_argument('--enable_feature_engineering', action='store_true', help='启用特征工程（仅非传统模式）')
    
    args = parser.parse_args()
    
    if args.use_traditional:
        summary = preprocess_data(
            sample_size=args.sample_size,
            window_size=args.window_size,
            test_sample_size=args.test_sample_size,
            save_dir=args.save_dir
        )
        print("传统数据预处理成功完成!")
    else:
        summary = preprocess_data_with_feature_engineering(
            sample_size=args.sample_size,
            window_size=args.window_size,
            test_sample_size=args.test_sample_size,
            save_dir=args.save_dir,
            enable_feature_engineering=args.enable_feature_engineering
        )
        print("量化金融背景下的特征工程数据预处理成功完成!")
    
    print(f"处理结果: {summary['summary_stats']}")
