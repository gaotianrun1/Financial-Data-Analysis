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
            # 计算预测价格的趋势变化
            predicted_trend = (predictions[i+1] - predictions[i]) / predictions[i]
            
            # 只有预测趋势向上且超过阈值才买入
            if predicted_trend > self.threshold:
                signals[i] = 1  # 全仓买入股票
            else:  # 预测趋势向下或涨幅不足
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
    
    def plot_backtest_results(self, backtest_result, save_dir=None, actual_prices=None, predictions=None):
        """绘制回测结果图表，分别保存多个图表"""
        strategy_returns = backtest_result['strategy_returns']
        portfolio_history = backtest_result['portfolio_history']
        signals = backtest_result['signals']
        
        if save_dir is None:
            save_dir = "."
        
        # 1. 价格预测和交易信号图（大图，重点）
        self._plot_price_signals(actual_prices, predictions, signals, f"{save_dir}/01_price_signals.png")
        
        # 2. 组合价值和回撤分析
        self._plot_portfolio_performance(portfolio_history, f"{save_dir}/02_portfolio_performance.png")
        
        # 3. 收益分析
        self._plot_returns_analysis(strategy_returns, f"{save_dir}/03_returns_analysis.png")
        
        # 4. 持仓分析
        self._plot_position_analysis(signals, f"{save_dir}/04_position_analysis.png")
        
        print(f"所有图表已保存到目录: {save_dir}")
    
    def _plot_price_signals(self, actual_prices, predictions, signals, save_path):
        """绘制价格预测和交易信号图，包含局部放大图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 14))
        
        # 添加基准价格
        actual_with_base = actual_prices + self.price
        pred_with_base = predictions + self.price
        
        # 标记买卖点
        buy_points = []
        sell_points = []
        current_position = 0
        
        for i, signal in enumerate(signals):
            if signal != current_position:
                if signal == 1:  # 买入
                    buy_points.append((i, actual_with_base[i]))
                else:  # 卖出
                    sell_points.append((i, actual_with_base[i]))
                current_position = signal
        
        # 第一个子图：完整数据
        time_index = range(len(actual_with_base))
        ax1.plot(time_index, actual_with_base, linewidth=1.5, color='#2E86AB', 
                label='Actual Price', alpha=0.9)
        ax1.plot(time_index, pred_with_base, linewidth=1.5, color='#E74C3C', 
                label='Predicted Price', alpha=0.8, linestyle='--')
        
        # 绘制买卖点
        if buy_points:
            buy_x, buy_y = zip(*buy_points)
            ax1.scatter(buy_x, buy_y, color='green', marker='^', s=80, 
                       label=f'Buy Signals ({len(buy_points)})', zorder=5, alpha=0.8)
        
        if sell_points:
            sell_x, sell_y = zip(*sell_points)
            ax1.scatter(sell_x, sell_y, color='red', marker='v', s=80, 
                       label=f'Sell Signals ({len(sell_points)})', zorder=5, alpha=0.8)
        
        ax1.set_title('Price Prediction & Trading Signals (Full Data)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        total_trades = len(buy_points) + len(sell_points)
        ax1.text(0.02, 0.98, f'Total Trades: {total_trades}\nThreshold: {self.threshold:.1%}\nCommission: {self.commission:.2%}', 
                transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 第二个子图：前两个小时的局部放大图
        zoom_samples = 120
        zoom_end = min(zoom_samples, len(actual_with_base))
        
        time_index_zoom = range(zoom_end)
        ax2.plot(time_index_zoom, actual_with_base[:zoom_end], linewidth=2.5, color='#2E86AB', 
                label='Actual Price', alpha=0.9, marker='o', markersize=4)
        ax2.plot(time_index_zoom, pred_with_base[:zoom_end], linewidth=2.5, color='#E74C3C', 
                label='Predicted Price', alpha=0.8, linestyle='--', marker='s', markersize=3)
        
        # 绘制局部买卖点
        zoom_buy_points = [(x, y) for x, y in buy_points if x < zoom_end]
        zoom_sell_points = [(x, y) for x, y in sell_points if x < zoom_end]
        
        if zoom_buy_points:
            buy_x_zoom, buy_y_zoom = zip(*zoom_buy_points)
            ax2.scatter(buy_x_zoom, buy_y_zoom, color='green', marker='^', s=150, 
                       label=f'Buy Signals ({len(zoom_buy_points)})', zorder=5, alpha=0.9)
        
        if zoom_sell_points:
            sell_x_zoom, sell_y_zoom = zip(*zoom_sell_points)
            ax2.scatter(sell_x_zoom, sell_y_zoom, color='red', marker='v', s=150, 
                       label=f'Sell Signals ({len(zoom_sell_points)})', zorder=5, alpha=0.9)
        
        ax2.set_title(f'Detailed View (First {zoom_end} Minutes)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Time (Minutes)', fontsize=12)
        ax2.set_ylabel('Price', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"价格信号图已保存: {save_path}")
    
    def _plot_portfolio_performance(self, portfolio_history, save_path):
        """绘制组合表现图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 累积收益曲线
        ax1.plot(portfolio_history, linewidth=2, color='#2E86AB')
        ax1.set_title('Portfolio Value', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        
        # 添加收益统计
        total_return = (portfolio_history[-1] - 1) * 100
        ax1.text(0.02, 0.98, f'Total Return: {total_return:.2f}%', 
               transform=ax1.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 回撤分析
        rolling_max = np.maximum.accumulate(portfolio_history)
        drawdowns = (portfolio_history - rolling_max) / rolling_max
        ax2.fill_between(range(len(drawdowns)), drawdowns, 0, 
                        color='#E74C3C', alpha=0.7)
        ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (Minutes)')
        ax2.set_ylabel('Drawdown Ratio')
        ax2.grid(True, alpha=0.3)
        
        # 添加最大回撤
        max_drawdown = np.min(drawdowns) * 100
        ax2.text(0.02, 0.02, f'Max Drawdown: {max_drawdown:.2f}%', 
               transform=ax2.transAxes, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"组合表现图已保存: {save_path}")
    
    def _plot_returns_analysis(self, strategy_returns, save_path):
        """绘制收益分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 只分析非零收益（即持股期间的收益）
        non_zero_returns = strategy_returns[strategy_returns != 0]
        
        if len(non_zero_returns) > 0:
            # 持股期间收益分布
            ax1.hist(non_zero_returns, bins=30, alpha=0.7, color='#27AE60', edgecolor='black', linewidth=0.5)
            ax1.axvline(np.mean(non_zero_returns), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(non_zero_returns):.4f}')
            ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
            
            # 添加统计信息
            positive_pct = np.sum(non_zero_returns > 0) / len(non_zero_returns) * 100
            ax1.text(0.02, 0.98, f'Win Rate: {positive_pct:.1f}%\nTotal Trades: {len(non_zero_returns)}', 
                    transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax1.set_title('Trading Returns Distribution\n(Holding Periods Only)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Returns')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No Trading Returns', ha='center', va='center', transform=ax1.transAxes)
        
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
            
            ax2.plot(rolling_sharpe, color='#8E44AD', linewidth=2)
            ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title(f'Rolling Sharpe Ratio\n{window_size}h window', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"收益分析图已保存: {save_path}")
    
    def _plot_position_analysis(self, signals, save_path):
        """绘制持仓分析图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 持仓时间分布
        hold_periods = []
        current_period = 1
        current_position = signals[0]
        
        for i in range(1, len(signals)):
            if signals[i] == current_position:
                current_period += 1
            else:
                hold_periods.append((current_position, current_period))
                current_position = signals[i]
                current_period = 1
        hold_periods.append((current_position, current_period))
        
        stock_periods = [p[1] for p in hold_periods if p[0] == 1]
        cash_periods = [p[1] for p in hold_periods if p[0] == 0]
        
        if stock_periods and cash_periods:
            ax1.hist([stock_periods, cash_periods], bins=20, alpha=0.7, 
                    label=['Stock Holding', 'Cash Holding'], 
                    color=['green', 'orange'], edgecolor='black', linewidth=0.5)
            ax1.set_title('Holding Period Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Holding Period (Minutes)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 持仓比例饼图
        stock_time = np.sum(signals == 1)
        cash_time = np.sum(signals == 0)
        
        labels = ['Stock Position', 'Cash Position']
        sizes = [stock_time, cash_time]
        colors = ['green', 'orange']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Position Time Allocation', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"持仓分析图已保存: {save_path}")


if __name__ == "__main__":
    print("开始加载训练结果并进行回测评估")
    print("=" * 60)

    model_dir = 'outputs/result_lstm'
    pred_file = f'{model_dir}/val_predictions_lstm.csv'

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
        commission=2/10000, # 交易手续费
        threshold=0.01  # 涨幅阈值
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
            save_dir=backtest_output_dir,
            actual_prices=actual_prices,
            predictions=predictions
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
        actual_with_base = actual_prices + backtester.price
        buy_hold_return = (actual_with_base[-1] / actual_with_base[0] - 1)
        strategy_return = results['metrics']['total_return']
        
        print(f"买入持有收益: {buy_hold_return:.2%}")
        print(f"策略总收益: {strategy_return:.2%}")
        print(f"超额收益: {(strategy_return - buy_hold_return):.2%}")

        print(f"\n 回测评估完成！")
        
    except Exception as e:
        print(f"回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
