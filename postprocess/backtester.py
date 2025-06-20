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
    def __init__(self, price=30, commission=0.001, threshold=0.005):
        self.commission = commission
        self.price = price
        self.threshold = threshold  # 涨幅阈值，只有预测涨幅超过此值才买入
        
    def minutely_strategy_backtest(self, predictions, actual_prices):
        """
        策略逻辑：每分钟判断下一分钟涨幅，预测涨幅>阈值：全仓买入股票，否则：全仓持有现金
        """
        n_periods = len(actual_prices) - 1
        
        actual_prices = actual_prices + self.price
        predictions = predictions + self.price
        
        # 生成交易信号
        signals = np.zeros(n_periods)
        for i in range(n_periods):
            # 计算预测的涨幅
            predicted_return = (predictions[i+1] - actual_prices[i]) / actual_prices[i]
            
            # 只有预测涨幅超过阈值才买入
            if predicted_return > self.threshold:
                signals[i] = 1  # 全仓买入股票
            else:  # 预测涨幅不足或会跌
                signals[i] = 0  # 持现金
        
        # 计算每分钟的价格收益率
        minutely_returns = np.diff(actual_prices) / actual_prices[:-1]
        # minutely_returns = np.clip(minutely_returns, -0.5, 0.5) # 限制单期收益率在[-50%, 50%]范围内
        
        # 逐步计算策略收益，考虑复利和交易成本
        portfolio_value = 1.0  # 初始价值为1（100%）
        portfolio_history = [portfolio_value]
        strategy_returns = []
        
        current_position = 0  # 当前仓位：0=持现金，1=持股票
        total_trades = 0
        trade_cost_total = 0.0
        
        for i in range(n_periods):
            target_position = signals[i]
            
            if target_position != current_position:
                # 发生交易，扣除交易成本
                trade_cost_amount = portfolio_value * self.commission  # 实际扣除的金额
                portfolio_value -= trade_cost_amount  # 直接扣除金额
                total_trades += 1
                trade_cost_total += trade_cost_amount  # 累加实际扣除的金额
                current_position = target_position
            
            # 计算当期收益
            if current_position == 1:  # 持股
                period_return = minutely_returns[i]
            else:  # 持现金
                period_return = 0.0
            
            # 更新价值
            portfolio_value *= (1 + period_return)
            portfolio_history.append(portfolio_value)
            
            # 计算策略收益率
            strategy_return = period_return if current_position == 1 else 0.0
            strategy_returns.append(strategy_return)
        
        strategy_returns = np.array(strategy_returns)
        
        buy_hours = np.sum(signals == 1)
        cash_hours = np.sum(signals == 0)
        
        return {
            'strategy_returns': strategy_returns,
            'minutely_returns': minutely_returns,
            'signals': signals,
            'total_trades': total_trades,
            'trade_cost_total': trade_cost_total,
            'buy_hours': buy_hours,
            'cash_hours': cash_hours,
            'portfolio_history': np.array(portfolio_history),
            'final_portfolio_value': portfolio_value
        }
    
    def calculate_performance_metrics(self, backtest_result, trading_freq='minutely'):
        """
        计算性能指标，支持不同交易频率
        
        Args:
            backtest_result: 回测结果字典
            trading_freq: 交易频率 ('minutely', 'daily')
        """
        portfolio_history = backtest_result['portfolio_history']
        strategy_returns = backtest_result['strategy_returns']
        
        # 根据交易频率确定年化系数
        if trading_freq == 'minutely':
            annual_factor = 365 * 24 * 60
        else:  # daily
            annual_factor = 252

        total_return = portfolio_history[-1] - 1.0  # 总收益率 = 最终价值 - 初始价值

        # 年化收益率
        n_periods = len(strategy_returns)
        annualized_return = (1 + total_return) ** (annual_factor / n_periods) - 1
            
        # 年化波动率
        volatility = np.std(strategy_returns) * np.sqrt(annual_factor)
        
        # 夏普比率(假设无风险利率为1%)，衡量每单位风险获得的超额收益
        risk_free_rate = 0.01
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 最大回撤
        rolling_max = np.maximum.accumulate(portfolio_history)
        drawdowns = (portfolio_history - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_periods': n_periods,
            'final_portfolio_value': portfolio_history[-1]
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
        metrics = self.calculate_performance_metrics(backtest_result, 'minutely')
        
        print(f"{'='*50}")
        
        print(f"\n 基础统计:")
        print(f"总时间段数: {metrics['total_periods']}")
        print(f"持股时间段: {backtest_result['buy_hours']} ({backtest_result['buy_hours']/metrics['total_periods']*100:.1f}%)")
        print(f"持现金时间段: {backtest_result['cash_hours']} ({backtest_result['cash_hours']/metrics['total_periods']*100:.1f}%)")
        print(f"总交易次数: {backtest_result['total_trades']}")
        print(f"总交易成本: {backtest_result['trade_cost_total']:.4f} ({backtest_result['trade_cost_total']*100:.2f}%)")
        
        print(f"\n 收益指标:")
        print(f"最终价值: {metrics['final_portfolio_value']:.4f}")
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
    
    def plot_backtest_results(self, backtest_result, save_path=None):
        """绘制回测结果图表"""
        strategy_returns = backtest_result['strategy_returns']
        portfolio_history = backtest_result['portfolio_history']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Simple Strategy Backtest Results', fontsize=14, fontweight='bold')
        
        # 累积收益曲线
        axes[0, 0].plot(portfolio_history, linewidth=2, color='#2E86AB')
        axes[0, 0].set_title('Portfolio Value')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 回撤分析
        rolling_max = np.maximum.accumulate(portfolio_history)
        drawdowns = (portfolio_history - rolling_max) / rolling_max
        axes[0, 1].fill_between(range(len(drawdowns)), drawdowns, 0, 
                               color='#E74C3C', alpha=0.7)
        axes[0, 1].set_title('Drawdown Analysis')
        axes[0, 1].set_ylabel('Drawdown Ratio') # 回撤比例
        axes[0, 1].grid(True, alpha=0.3)
        
        # 收益分布
        axes[1, 0].hist(strategy_returns, bins=50, alpha=0.7, color='#27AE60')
        axes[1, 0].axvline(np.mean(strategy_returns), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(strategy_returns):.4f}')
        axes[1, 0].set_title('Returns Distribution')
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
            axes[1, 1].set_title(f'Rolling Sharpe Ratio ({window_size}h window)')
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

    # 创建回测器
    backtester = SimpleBacktester(
        price=30,        # 假设基准价格30元
        commission=2/10000, # 0.03% 交易手续费
        threshold=0.001  # 涨幅阈值
    )
    
    try:
        results = backtester.run_test_evaluation(
            predictions=predictions,
            actual_prices=actual_prices,
        )

        backtest_output_dir = f"{model_dir}/backtest_results"
        os.makedirs(backtest_output_dir, exist_ok=True)
        
        backtester.plot_backtest_results(
            results['backtest_result'],
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
        
        print(f"\n 与买入持有策略对比:")
        buy_hold_return = (actual_prices[-1] / actual_prices[0] - 1)
        strategy_return = results['metrics']['total_return']
        
        print(f"买入持有收益: {buy_hold_return:.2%}")
        print(f"策略总收益: {strategy_return:.2%}")
        print(f"超额收益: {(strategy_return - buy_hold_return):.2%}")

        print(f"\n 回测评估完成！")
        
    except Exception as e:
        print(f"回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
