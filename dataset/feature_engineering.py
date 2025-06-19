"""
量化金融特征工程统一模块 - 针对高频交易数据优化
基于bid_qty, ask_qty, buy_qty, sell_qty, volume构造专业量化特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class IntegratedFeatureEngineer:
    """集成特征工程器 - 整合所有特征工程和预处理方法"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_engineer_config = self.config.get('feature_engineering', {})
        
        # 存储生成的特征名称
        self.order_flow_features = []      # 订单流特征
        self.liquidity_features = []       # 流动性特征
        self.microstructure_features = []  # 微观结构特征
        self.pressure_features = []        # 买卖压力特征
        self.statistical_features = []     # 统计特征
        self.time_features = []           # 时间特征
        self.interaction_features = []     # 交互特征
        self.anonymous_features = []       # 匿名特征处理
        
        # 存储处理器
        self.scalers = {}
        self.feature_selectors = {}
        
        # 统计信息
        self.processing_stats = {}
        
    def apply_feature_engineering(self, df: pd.DataFrame, target_col: str = 'label') -> pd.DataFrame:
        """应用综合特征工程"""
        print(f"开始应用量化金融特征工程...")
        print(f"输入数据形状: {df.shape}")
        
        original_features = df.shape[1]
        df_processed = df.copy()
        
        # 1. 订单流特征
        if self.feature_engineer_config.get('enable_order_flow_features', True):
            df_processed = self._add_order_flow_features(df_processed)
        
        # 2. 流动性特征
        if self.feature_engineer_config.get('enable_liquidity_features', True):
            df_processed = self._add_liquidity_features(df_processed)
        
        # 3. 微观结构特征
        if self.feature_engineer_config.get('enable_microstructure_features', True):
            df_processed = self._add_microstructure_features(df_processed)
        
        # 4. 买卖压力特征
        if self.feature_engineer_config.get('enable_pressure_features', True):
            df_processed = self._add_pressure_features(df_processed)
        
        # 5. 统计特征
        if self.feature_engineer_config.get('enable_statistical_features', True):
            df_processed = self._add_statistical_features(df_processed, target_col)
        
        # 6. 时间特征
        if self.feature_engineer_config.get('enable_time_features', True):
            df_processed = self._add_time_features(df_processed)
        
        # 7. 交互特征
        if self.feature_engineer_config.get('enable_interaction_features', True):
            df_processed = self._add_interaction_features(df_processed)
        
        # 8. 特征工程后的数据清洗
        print("\n特征工程完成，开始数据清洗...")
        df_processed = self._clean_engineered_features(df_processed, target_col)
        
        new_features = df_processed.shape[1] - original_features
        print(f"特征工程完成: 原始{original_features}个特征 -> 现在{df_processed.shape[1]}个特征 (新增{new_features}个)")
        
        return df_processed
    
    def _clean_engineered_features(self, df: pd.DataFrame, target_col: str = 'label') -> pd.DataFrame:
        """
        使用data_loader中的清洗函数处理特征工程后的数据
        """
        print("清理特征工程后的数据...")
        from .data_loader import handle_missing_values, handle_financial_outliers
        
        # 处理缺失值和无穷值
        df_clean, removed_features = handle_missing_values(
            df, 
            missing_threshold=0.5,  # 对于特征工程后的数据，使用更宽松的阈值
            method='ffill'
        )
        
        # 处理异常值
        df_clean, outlier_info = handle_financial_outliers(
            df_clean,
            method='iqr',
            window='1D',
            target_col=target_col
        )
        
        print(f"  数据清洗完成: {df.shape} -> {df_clean.shape}")
        return df_clean
    
    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加订单流特征 - 量化交易的核心特征"""
        print("添加订单流特征...")
        df_enhanced = df.copy()
        
        # 基础订单流不平衡指标
        # OFI (Order Flow Imbalance) - 衡量买卖订单的不平衡程度
        df_enhanced['order_flow_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        
        # 交易流不平衡 - 衡量主动买卖的不平衡
        df_enhanced['trade_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
        
        # 净订单流 - 绝对数量差异
        df_enhanced['net_order_flow'] = df['bid_qty'] - df['ask_qty']
        df_enhanced['net_trade_flow'] = df['buy_qty'] - df['sell_qty']
        
        # 订单流强度
        df_enhanced['order_flow_intensity'] = df['bid_qty'] + df['ask_qty']
        df_enhanced['trade_flow_intensity'] = df['buy_qty'] + df['sell_qty']
        
        self.order_flow_features.extend([
            'order_flow_imbalance', 'trade_flow_imbalance', 'net_order_flow', 
            'net_trade_flow', 'order_flow_intensity', 'trade_flow_intensity'
        ])
        
        # 滚动订单流特征
        windows = self.feature_engineer_config.get('order_flow_windows', [5, 10, 20])
        for window in windows:
            # 滚动订单流不平衡均值
            df_enhanced[f'order_flow_imbalance_ma_{window}'] = df_enhanced['order_flow_imbalance'].rolling(window=window, min_periods=1).mean()
            df_enhanced[f'trade_flow_imbalance_ma_{window}'] = df_enhanced['trade_flow_imbalance'].rolling(window=window, min_periods=1).mean()
            
            # 滚动标准差 - 衡量订单流波动性
            df_enhanced[f'order_flow_imbalance_std_{window}'] = df_enhanced['order_flow_imbalance'].rolling(window=window, min_periods=1).std()
            df_enhanced[f'trade_flow_imbalance_std_{window}'] = df_enhanced['trade_flow_imbalance'].rolling(window=window, min_periods=1).std()
            
            # 订单流动量指标
            df_enhanced[f'order_flow_momentum_{window}'] = df_enhanced['order_flow_imbalance'] - df_enhanced[f'order_flow_imbalance_ma_{window}']
            df_enhanced[f'trade_flow_momentum_{window}'] = df_enhanced['trade_flow_imbalance'] - df_enhanced[f'trade_flow_imbalance_ma_{window}']
            
            self.order_flow_features.extend([
                f'order_flow_imbalance_ma_{window}', f'trade_flow_imbalance_ma_{window}',
                f'order_flow_imbalance_std_{window}', f'trade_flow_imbalance_std_{window}',
                f'order_flow_momentum_{window}', f'trade_flow_momentum_{window}'
            ])
        
        print(f"    添加了 {len([f for f in self.order_flow_features if f in df_enhanced.columns])} 个订单流特征")
        return df_enhanced
    
    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加流动性特征"""
        print("  添加流动性特征...")
        df_enhanced = df.copy()
        
        # 流动性指标
        # 总流动性
        df_enhanced['total_liquidity'] = df['bid_qty'] + df['ask_qty']
        
        # 流动性不对称性
        df_enhanced['liquidity_asymmetry'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        
        # 流动性消耗率 - 成交量与挂单量的比率
        df_enhanced['liquidity_consumption_rate'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        
        # 买卖流动性比率
        df_enhanced['bid_ask_liquidity_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-10)
        
        # 主动交易与被动流动性的比率
        df_enhanced['active_passive_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        
        self.liquidity_features.extend([
            'total_liquidity', 'liquidity_asymmetry', 'liquidity_consumption_rate',
            'bid_ask_liquidity_ratio', 'active_passive_ratio'
        ])
        
        # 滚动流动性特征
        windows = self.feature_engineer_config.get('liquidity_windows', [5, 10, 20])
        for window in windows:
            # 平均流动性
            df_enhanced[f'avg_liquidity_{window}'] = df_enhanced['total_liquidity'].rolling(window=window, min_periods=1).mean()
            
            # 流动性波动率
            df_enhanced[f'liquidity_volatility_{window}'] = df_enhanced['total_liquidity'].rolling(window=window, min_periods=1).std()
            
            # 流动性消耗率均值
            df_enhanced[f'avg_liquidity_consumption_{window}'] = df_enhanced['liquidity_consumption_rate'].rolling(window=window, min_periods=1).mean()
            
            self.liquidity_features.extend([
                f'avg_liquidity_{window}', f'liquidity_volatility_{window}', f'avg_liquidity_consumption_{window}'
            ])
        
        print(f"添加了 {len([f for f in self.liquidity_features if f in df_enhanced.columns])} 个流动性特征")
        return df_enhanced
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加市场微观结构特征"""
        print("添加市场微观结构特征...")
        df_enhanced = df.copy()
        
        # 价格影响指标
        # 买卖价差代理（使用数量差异）
        df_enhanced['spread_proxy'] = np.abs(df['bid_qty'] - df['ask_qty'])
        
        # 市场深度
        df_enhanced['market_depth'] = np.minimum(df['bid_qty'], df['ask_qty'])
        
        # 订单簿斜率 - 衡量流动性分布
        df_enhanced['orderbook_slope'] = (df['bid_qty'] + df['ask_qty']) / (np.abs(df['bid_qty'] - df['ask_qty']) + 1e-10)
        
        # 执行概率 - 基于订单大小和可用流动性
        df_enhanced['execution_probability'] = np.minimum(df['buy_qty'] / (df['ask_qty'] + 1e-10), 1.0)
        
        # 市场冲击指标
        df_enhanced['market_impact'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        
        # 信息比率 - 永久价格影响代理
        df_enhanced['information_ratio'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
        
        self.microstructure_features.extend([
            'spread_proxy', 'market_depth', 'orderbook_slope', 
            'execution_probability', 'market_impact', 'information_ratio'
        ])
        
        # 滚动微观结构特征
        windows = self.feature_engineer_config.get('microstructure_windows', [5, 10, 20])
        for window in windows:
            # 滚动价差
            df_enhanced[f'avg_spread_proxy_{window}'] = df_enhanced['spread_proxy'].rolling(window=window, min_periods=1).mean()
            
            # 滚动市场深度
            df_enhanced[f'avg_market_depth_{window}'] = df_enhanced['market_depth'].rolling(window=window, min_periods=1).mean()
            
            # 滚动市场冲击
            df_enhanced[f'avg_market_impact_{window}'] = df_enhanced['market_impact'].rolling(window=window, min_periods=1).mean()
            
            self.microstructure_features.extend([
                f'avg_spread_proxy_{window}', f'avg_market_depth_{window}', f'avg_market_impact_{window}'
            ])
        
        print(f"    添加了 {len([f for f in self.microstructure_features if f in df_enhanced.columns])} 个微观结构特征")
        return df_enhanced
    
    def _add_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加买卖压力特征"""
        print("  添加买卖压力特征...")
        df_enhanced = df.copy()
        
        # 买入压力指标
        df_enhanced['buy_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
        df_enhanced['sell_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
        
        # 净买入压力
        df_enhanced['net_buy_pressure'] = df_enhanced['buy_pressure'] - df_enhanced['sell_pressure']
        
        # 买卖强度比
        df_enhanced['buy_sell_intensity_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
        
        # 挂单压力
        df_enhanced['bid_pressure'] = df['bid_qty'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        df_enhanced['ask_pressure'] = df['ask_qty'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        
        # 压力不平衡
        df_enhanced['pressure_imbalance'] = df_enhanced['bid_pressure'] - df_enhanced['ask_pressure']
        
        # 交易压力与挂单压力的关系
        df_enhanced['trading_vs_quoting_pressure'] = (df['buy_qty'] + df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        
        self.pressure_features.extend([
            'buy_pressure', 'sell_pressure', 'net_buy_pressure', 'buy_sell_intensity_ratio',
            'bid_pressure', 'ask_pressure', 'pressure_imbalance', 'trading_vs_quoting_pressure'
        ])
        
        # 滚动压力特征
        windows = self.feature_engineer_config.get('pressure_windows', [5, 10, 20])
        for window in windows:
            # 滚动买卖压力
            df_enhanced[f'avg_buy_pressure_{window}'] = df_enhanced['buy_pressure'].rolling(window=window, min_periods=1).mean()
            df_enhanced[f'avg_sell_pressure_{window}'] = df_enhanced['sell_pressure'].rolling(window=window, min_periods=1).mean()
            df_enhanced[f'avg_net_buy_pressure_{window}'] = df_enhanced['net_buy_pressure'].rolling(window=window, min_periods=1).mean()
            
            # 压力波动率
            df_enhanced[f'buy_pressure_volatility_{window}'] = df_enhanced['buy_pressure'].rolling(window=window, min_periods=1).std()
            df_enhanced[f'sell_pressure_volatility_{window}'] = df_enhanced['sell_pressure'].rolling(window=window, min_periods=1).std()
            
            self.pressure_features.extend([
                f'avg_buy_pressure_{window}', f'avg_sell_pressure_{window}', f'avg_net_buy_pressure_{window}',
                f'buy_pressure_volatility_{window}', f'sell_pressure_volatility_{window}'
            ])
        
        print(f"    添加了 {len([f for f in self.pressure_features if f in df_enhanced.columns])} 个买卖压力特征")
        return df_enhanced

    def _add_statistical_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """添加统计特征"""
        print("  添加统计特征...")
        df_enhanced = df.copy()
        
        # 基础市场数据列
        key_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        
        # 滚动窗口统计
        stat_windows = self.feature_engineer_config.get('statistical_windows', [5, 10, 20])
        operations = ['mean', 'std', 'max', 'min']
        
        for col in key_cols:
            for window in stat_windows:
                for op in operations:
                    feature_name = f'{col}_{op}_{window}'
                    if op == 'mean':
                        df_enhanced[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                    elif op == 'std':
                        df_enhanced[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                    elif op == 'max':
                        df_enhanced[feature_name] = df[col].rolling(window=window, min_periods=1).max()
                    elif op == 'min':
                        df_enhanced[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                    
                    self.statistical_features.append(feature_name)
                
                # Z-score特征 - 标准化偏离度
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                rolling_std = df[col].rolling(window=window, min_periods=1).std()
                z_score_name = f'{col}_zscore_{window}'
                rolling_std_safe = np.maximum(rolling_std, 1e-8)
                df_enhanced[z_score_name] = (df[col] - rolling_mean) / rolling_std_safe
                self.statistical_features.append(z_score_name)
                
                # 相对变化率
                pct_change_name = f'{col}_pct_change_{window}'
                df_enhanced[pct_change_name] = df[col].pct_change(periods=window)
                self.statistical_features.append(pct_change_name)
        
        # 滞后特征 - 重点关注标签的历史值
        lag_periods = self.feature_engineer_config.get('lag_periods', [1, 2, 3, 5, 10])
        if target_col in df.columns:
            for lag in lag_periods:
                lag_name = f'{target_col}_lag_{lag}'
                df_enhanced[lag_name] = df[target_col].shift(lag)
                self.statistical_features.append(lag_name)
        
        # 市场数据的滞后特征
        for col in ['volume']:
            for lag in [1, 2, 3]:
                lag_name = f'{col}_lag_{lag}'
                df_enhanced[lag_name] = df[col].shift(lag)
                self.statistical_features.append(lag_name)
        
        # 差分特征
        for col in key_cols:
            df_enhanced[f'{col}_diff_1'] = df[col].diff(1)
            df_enhanced[f'{col}_diff_2'] = df[col].diff(2)
            self.statistical_features.extend([f'{col}_diff_1', f'{col}_diff_2'])
        
        print(f"添加了 {len([f for f in self.statistical_features if f in df_enhanced.columns])} 个统计特征")
        return df_enhanced
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        print("  添加时间特征...")
        df_enhanced = df.copy()

        # 基础时间特征，捕捉交易时间的周期性规律
        df_enhanced['hour'] = df_enhanced.index.hour
        df_enhanced['minute'] = df_enhanced.index.minute
        df_enhanced['day_of_week'] = df_enhanced.index.dayofweek
        df_enhanced['month'] = df_enhanced.index.month
        
        basic_time_features = ['hour', 'minute', 'day_of_week', 'month']
        self.time_features.extend(basic_time_features)
        
        # 周期性编码，保持时间的周期性特性，避免边界问题
        if self.feature_engineer_config.get('enable_cyclical_encoding', True):
            df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
            df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
            df_enhanced['minute_sin'] = np.sin(2 * np.pi * df_enhanced['minute'] / 60)
            df_enhanced['minute_cos'] = np.cos(2 * np.pi * df_enhanced['minute'] / 60)
            df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
            df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
            
            cyclical_features = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos']
            self.time_features.extend(cyclical_features)
        
        # 交易时段特征，似乎没看到休市
        
        print(f"添加了 {len(self.time_features)} 个时间特征")
        return df_enhanced
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交互特征"""
        print("  添加交互特征...")
        df_enhanced = df.copy()
        # 买卖量与流动性的交互
        df_enhanced['buy_liquidity_interaction'] = df['buy_qty'] * df['bid_qty']
        df_enhanced['sell_liquidity_interaction'] = df['sell_qty'] * df['ask_qty']
        
        # 订单流与成交量的交互
        df_enhanced['order_flow_volume_interaction'] = (df['bid_qty'] - df['ask_qty']) * df['volume']
        df_enhanced['trade_flow_volume_interaction'] = (df['buy_qty'] - df['sell_qty']) * df['volume']
        
        # 压力指标的交互
        if 'buy_pressure' in df_enhanced.columns and 'liquidity_consumption_rate' in df_enhanced.columns:
            df_enhanced['pressure_liquidity_interaction'] = df_enhanced['buy_pressure'] * df_enhanced['liquidity_consumption_rate']
        
        # 时间与市场数据的交互
        if 'hour' in df_enhanced.columns:
            df_enhanced['volume_hour_interaction'] = df['volume'] * df_enhanced['hour']
            df_enhanced['order_flow_hour_interaction'] = (df['bid_qty'] - df['ask_qty']) * df_enhanced['hour']
        
        self.interaction_features.extend([
            'buy_liquidity_interaction', 'sell_liquidity_interaction',
            'order_flow_volume_interaction', 'trade_flow_volume_interaction',
            'volume_hour_interaction', 'order_flow_hour_interaction'
        ])
        
        print(f"    添加了 {len([f for f in self.interaction_features if f in df_enhanced.columns])} 个交互特征")
        return df_enhanced
  
    def get_feature_summary(self) -> Dict:
        """获取特征工程总结"""
        return {
            'order_flow_features': len(self.order_flow_features),
            'liquidity_features': len(self.liquidity_features),
            'microstructure_features': len(self.microstructure_features),
            'pressure_features': len(self.pressure_features),
            'statistical_features': len(self.statistical_features),
            'time_features': len(self.time_features),
            'interaction_features': len(self.interaction_features),
            'anonymous_features': len(self.anonymous_features),
            'total_new_features': (len(self.order_flow_features) + len(self.liquidity_features) + 
                                 len(self.microstructure_features) + len(self.pressure_features) +
                                 len(self.statistical_features) + len(self.time_features) + 
                                 len(self.interaction_features) + len(self.anonymous_features)),
            'feature_names': {
                'order_flow_features': self.order_flow_features,
                'liquidity_features': self.liquidity_features,
                'microstructure_features': self.microstructure_features,
                'pressure_features': self.pressure_features,
                'statistical_features': self.statistical_features,
                'time_features': self.time_features,
                'interaction_features': self.interaction_features,
                'anonymous_features': self.anonymous_features
            }
        }

# 主函数
def apply_integrated_feature_engineering(df: pd.DataFrame, config: Dict, target_col: str = 'label') -> Tuple[pd.DataFrame, Dict]:
    """
    应用量化金融特征工程流程
    
    Args:
        df: 输入数据
        config: 配置字典
        target_col: 目标列名
        
    Returns:
        处理后的数据框和处理信息
    """
    engineer = IntegratedFeatureEngineer(config)
    
    # 特征工程
    df_processed = engineer.apply_feature_engineering(df, target_col)
    
    # 整合信息
    processing_info = {
        'feature_summary': engineer.get_feature_summary(),
        'scalers': engineer.scalers,
        'original_shape': df.shape,
        'final_shape': df_processed.shape
    }
    
    return df_processed, processing_info 