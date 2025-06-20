# python postprocess/backtester.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import glob
import os
import json

class SimpleBacktester:
    def __init__(self, price=30, commission=0.001):
        self.commission = commission
        self.price = price
        
    def minutely_strategy_backtest(self, predictions, actual_prices):
        """
        策略逻辑：每分钟判断下一分钟涨跌，预测涨：全仓买入股票，预测跌：全仓持有现金
        """
        n_periods = len(actual_prices) - 1
        actual_prices = actual_prices + self.price
        predictions = predictions + self.price
        
        # 生成交易信号
        signals = np.zeros(n_periods)
        for i in range(n_periods):
            # 判断下一分钟会涨还是跌
            if predictions[i+1] > actual_prices[i]:  # 会涨
                signals[i] = 1  # 全仓买入股票
            else:  # 会跌或持平
                signals[i] = 0  # 持现金
        
        # 计算每分钟的价格收益率
        minutely_returns = np.diff(actual_prices) / actual_prices[:-1]
        # minutely_returns = np.clip(minutely_returns, -0.5, 0.5) # 限制单期收益率在[-50%, 50%]范围内
        
        # 计算收益
        strategy_returns = signals * minutely_returns
        
        # 计算交易成本
        position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
        total_trades = np.sum(position_changes)
        trade_cost_total = total_trades * self.commission
        
        # 统计信息
        buy_hours = np.sum(signals == 1)
        cash_hours = np.sum(signals == 0)
        
        return {
            'strategy_returns': strategy_returns,
            'minutely_returns': minutely_returns,
            'signals': signals,
            'total_trades': total_trades,
            'trade_cost_total': trade_cost_total,
            'buy_hours': buy_hours,
            'cash_hours': cash_hours
        }
    
    def calculate_performance_metrics(self, strategy_returns, trading_freq='minutely'):
        """
        计算性能指标，支持不同交易频率
        
        Args:
            strategy_returns: 策略收益率序列
            trading_freq: 交易频率 ('minutely', 'daily')
        """
        # 根据交易频率确定年化系数
        if trading_freq == 'minutely':
            annual_factor = 365 * 24
        else:  # daily
            annual_factor = 252

        # 年化收益率
        total_return = np.prod(1 + strategy_returns) - 1
        n_periods = len(strategy_returns)
        annualized_return = (1 + total_return) ** (annual_factor / n_periods) - 1
            
        # 年化波动率
        volatility = np.std(strategy_returns) * np.sqrt(annual_factor)
        
        # 夏普比率(假设无风险利率为1%)，衡量每单位风险获得的超额收益
        risk_free_rate = 0.01
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 最大回撤，反映策略的风险承受能力
        cumulative_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_periods': n_periods
        }
    
    def run_test_evaluation(self, predictions, actual_prices, test_start_idx=None):
        """
        在测试集上运行评估
        
        Args:
            predictions: 完整的预测序列
            actual_prices: 完整的实际价格序列  
            test_start_idx: 测试集开始索引
        """
        backtest_result = self.minutely_strategy_backtest(predictions, actual_prices)
        
        # 计算性能指标
        metrics = self.calculate_performance_metrics(backtest_result['strategy_returns'], 'minutely')
        
        print(f"{'='*50}")
        
        print(f"\n 基础统计:")
        print(f"总时间段数: {metrics['total_periods']}")
        print(f"持股时间段: {backtest_result['buy_hours']} ({backtest_result['buy_hours']/metrics['total_periods']*100:.1f}%)")
        print(f"持现金时间段: {backtest_result['cash_hours']} ({backtest_result['cash_hours']/metrics['total_periods']*100:.1f}%)")
        print(f"总交易次数: {backtest_result['total_trades']}")
        print(f"交易成本: {backtest_result['trade_cost_total']:.4f}")
        
        print(f"\n 收益指标:")
        print(f"总收益率: {metrics['total_return']:.2%}")
        print(f"年化收益率: {metrics['annualized_return']:.2%}")
        print(f"年化波动率: {metrics['volatility']:.2%}")
        
        print(f"\n 风险调整指标:")
        print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        
        return {
            'backtest_result': backtest_result,
            'metrics': metrics,
            'test_start_idx': test_start_idx
        }
    
    def plot_backtest_results(self, strategy_returns, save_path=None):
        """绘制回测结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Simple Strategy Backtest Results', fontsize=14, fontweight='bold') # 简化策略回测结果
        
        # 累积收益曲线
        cumulative_returns = np.cumprod(1 + strategy_returns)
        axes[0, 0].plot(cumulative_returns, linewidth=2, color='#2E86AB')
        axes[0, 0].set_title('Cumulative Returns') # 累积收益曲线
        axes[0, 0].set_ylabel('Cumulative Returns') # 累积收益
        axes[0, 0].grid(True, alpha=0.3)
        
        # 回撤分析
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        axes[0, 1].fill_between(range(len(drawdowns)), drawdowns, 0, 
                               color='#E74C3C', alpha=0.7)
        axes[0, 1].set_title('Drawdown Analysis') # 回撤分析
        axes[0, 1].set_ylabel('Drawdown Ratio') # 回撤比例
        axes[0, 1].grid(True, alpha=0.3)
        
        # 收益分布
        axes[1, 0].hist(strategy_returns, bins=50, alpha=0.7, color='#27AE60')
        axes[1, 0].axvline(np.mean(strategy_returns), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(strategy_returns):.4f}')
        axes[1, 0].set_title('Returns Distribution') # 收益率分布
        axes[1, 0].set_xlabel('Returns') # 收益率
        axes[1, 0].set_ylabel('Frequency') # 频次
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 滚动夏普比率
        window_size = min(168, len(strategy_returns) // 4)
        if window_size > 10:
            rolling_sharpe = []
            for i in range(window_size, len(strategy_returns)):
                window_ret = strategy_returns[i-window_size:i]
                if np.std(window_ret) > 0:
                    sharpe = np.mean(window_ret) * np.sqrt(252*24) / (np.std(window_ret) * np.sqrt(252*24))
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(0)
            
            axes[1, 1].plot(rolling_sharpe, color='#8E44AD', linewidth=2)
            axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title(f'Rolling Sharpe Ratio ({window_size}h window)') # 滚动夏普比率
            axes[1, 1].set_ylabel('Sharpe Ratio') # 夏普比率
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")


if __name__ == "__main__":
    print("开始加载训练结果并进行回测评估")
    print("=" * 60)

    model_dir = 'outputs/result_transformer'
    pred_file = f'{model_dir}/val_predictions_transformer.csv'

    df = pd.read_csv(pred_file)
    print(f"成功加载预测数据，共 {len(df)} 个样本")
    
    actual_col = None
    pred_col = None
    
    for col in df.columns:
        if 'actual' in col.lower():
            actual_col = col
        elif 'pred' in col.lower():
            pred_col = col

    actual_prices = df[actual_col].values
    predictions = df[pred_col].values

    # 创建回测器并运行回测 
    backtester = SimpleBacktester(
        price=30,        # 假设基准价格30元
        commission=0.001 # 0.1% 交易手续费
    )
    
    try:
        # 运行回测评估
        results = backtester.run_test_evaluation(
            predictions=predictions,
            actual_prices=actual_prices,
            test_start_idx=0  # 使用全部数据，因为这已经是测试/验证集
        )
        
        # 保存回测结果
        backtest_output_dir = f"{model_dir}/backtest_results"
        os.makedirs(backtest_output_dir, exist_ok=True)
        
        # 绘制回测结果图表
        backtester.plot_backtest_results(
            results['backtest_result']['strategy_returns'],
            f"{backtest_output_dir}/simple_strategy_backtest.png"
        )
        
        # 处理NaN值和numpy数据类型，确保JSON可序列化
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        
        backtest_metrics = {
            'data_source': pred_file,
            'sample_count': len(actual_prices),
            'strategy_description': '分钟级简化策略：预测上涨全仓买入，预测下跌持现金',
            'commission_rate': backtester.commission,
            'metrics': convert_to_json_serializable(results['metrics']),
            'trading_stats': {
                'total_trades': int(results['backtest_result']['total_trades']),
                'buy_hours': int(results['backtest_result']['buy_hours']),
                'cash_hours': int(results['backtest_result']['cash_hours']),
                'trade_cost_total': float(results['backtest_result']['trade_cost_total'])
            }
        }
        
        metrics_file = f"{backtest_output_dir}/backtest_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(backtest_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 回测结果已保存到: {backtest_output_dir}/")
        print(f"   - 图表文件: simple_strategy_backtest.png")
        print(f"   - 指标文件: backtest_metrics.json")
        
        # 基准对比（买入并持有策略）
        print(f"\n与买入持有策略对比:")
        buy_hold_return = (actual_prices[-1] / actual_prices[0] - 1)
        strategy_return = results['metrics']['total_return']
        
        print(f"买入持有收益: {buy_hold_return:.2%}")
        print(f"策略总收益: {strategy_return:.2%}")
        print(f"超额收益: {(strategy_return - buy_hold_return):.2%}")
        
        if strategy_return > buy_hold_return:
            print("策略表现优于买入持有")
        else:
            print("策略表现不如买入持有")

        print(f"\n 回测评估完成！")
        
    except Exception as e:
        print(f"❌ 回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
