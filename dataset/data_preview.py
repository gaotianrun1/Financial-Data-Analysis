import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from datetime import datetime, timedelta

# Get the current script's directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "preview")

def preview_parquet(file_path):
    """Simple preview of parquet file format"""
    try:
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        print(f"File: {file_path}")
        print(f"Shape: {df.shape} (rows, columns)")
        print(f"Index type: {type(df.index).__name__}")
        
        print("\nColumn names:")
        print(df.columns.tolist())
        
        print("\nData types:")
        print(df.dtypes)
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nLast 5 rows:")
        print(df.tail())
        
        # Time range analysis
        print(f"\nTime range:")
        print(f"Start time: {df.index.min()}")
        print(f"End time: {df.index.max()}")
        print(f"Total duration: {df.index.max() - df.index.min()}")
        print(f"Data frequency: {pd.infer_freq(df.index[:1000])}")  # Infer frequency
        
        # print("\nBasic statistics:")
        # print(df.describe())
        
    except Exception as e:
        print(f"Failed to read file: {e}")

def aggregate_data(df, freq='H', agg_method='mean'):
    """
    Aggregate data to specified frequency
    
    Parameters:
    df: DataFrame with datetime index
    freq: str, aggregation frequency ('T'=minute, 'H'=hour, 'D'=day)
    agg_method: str, aggregation method ('mean', 'sum', 'max', 'min', 'last')
    
    Returns:
    aggregated DataFrame
    """
    # Use different aggregation methods for different columns
    agg_dict = {}
    
    # Volume-related columns use sum
    volume_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    for col in volume_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'
    
    # X features and label use mean
    feature_cols = [col for col in df.columns if col.startswith('X') or col == 'label']
    for col in feature_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    # Other columns use mean
    for col in df.columns:
        if col not in agg_dict:
            agg_dict[col] = 'mean'
    
    try:
        # Perform aggregation
        result = df.resample(freq).agg(agg_dict)
        
        # Instead of dropping all rows with any NaN, only drop rows where ALL values are NaN
        # This preserves rows that have some valid data
        result = result.dropna(how='all')
        
        # For remaining NaN values, use forward fill and backward fill
        result = result.fillna(method='ffill').fillna(method='bfill')
        
        return result
    except Exception as e:
        print(f"Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return df

def get_time_range_data(df, time_level='minute', specific_period=None):
    """
    Get data for specific time range based on time level
    
    Parameters:
    df: DataFrame with datetime index
    time_level: str, 'minute', 'hour', 'day'
    specific_period: str, specific time period in format:
        - minute level: 'YYYY-MM-DD' (one day)
        - hour level: 'YYYY-MM' (one month)  
        - day level: 'YYYY' (one year)
    
    Returns:
    filtered DataFrame and period description
    """
    if specific_period is None:
        # Automatically select representative time periods
        if time_level == 'minute':
            # Select the day with most data
            daily_counts = df.resample('D').size()
            best_day = daily_counts.idxmax().strftime('%Y-%m-%d')
            specific_period = best_day
        elif time_level == 'hour':
            # Select the month with most data
            monthly_counts = df.resample('M').size()
            best_month = monthly_counts.idxmax().strftime('%Y-%m')
            specific_period = best_month
        elif time_level == 'day':
            # Select the year with most data
            yearly_counts = df.resample('Y').size()
            best_year = yearly_counts.idxmax().strftime('%Y')
            specific_period = best_year
    
    # Filter data based on time level
    if time_level == 'minute':
        # Minute data for one day
        start_date = pd.to_datetime(specific_period)
        end_date = start_date + timedelta(days=1)
        filtered_df = df[start_date:end_date - timedelta(minutes=1)]
        period_desc = f"Minute-level data - {specific_period}"
        
    elif time_level == 'hour':
        # Hourly data for one month
        start_date = pd.to_datetime(specific_period)
        if start_date.month == 12:
            end_date = start_date.replace(year=start_date.year + 1, month=1)
        else:
            end_date = start_date.replace(month=start_date.month + 1)
        filtered_df = df[start_date:end_date - timedelta(hours=1)]
        # Aggregate to hourly level
        filtered_df = aggregate_data(filtered_df, freq='H')
        period_desc = f"Hourly data - {specific_period}"
        
    elif time_level == 'day':
        # Daily data for one year
        start_date = pd.to_datetime(specific_period + '-01-01')
        end_date = pd.to_datetime(specific_period + '-12-31') + timedelta(days=1)
        filtered_df = df[start_date:end_date - timedelta(days=1)]
        # Aggregate to daily level
        filtered_df = aggregate_data(filtered_df, freq='D')
        period_desc = f"Daily data - {specific_period}"
    
    else:
        raise ValueError("time_level must be 'minute', 'hour', or 'day'")
    
    return filtered_df, period_desc

def plot_time_series_multi_level(file_path, output_dir=None, 
                                 time_level='minute', specific_period=None):
    """
    Plot time series charts for specified time level and time range
    
    Parameters:
    file_path: str, path to parquet file
    output_dir: str, output directory
    time_level: str, 'minute', 'hour', 'day'
    specific_period: str, specific time period
    """
    try:
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        
        # Read parquet file
        print(f"Reading data file: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Ensure index is datetime type
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Get data for specified time range
        print(f"Processing {time_level}-level data...")
        filtered_df, period_desc = get_time_range_data(df, time_level, specific_period)
        
        print(f"Data range: {period_desc}")
        print(f"Data points: {len(filtered_df)}")
        
        if len(filtered_df) == 0:
            print("Warning: No data in the specified time range")
            return
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving charts to: {output_dir}")
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # Set different time formats based on time level
        if time_level == 'minute':
            time_format = '%H:%M'
            time_locator = mdates.HourLocator(interval=2)
        elif time_level == 'hour':
            time_format = '%m-%d'
            time_locator = mdates.DayLocator(interval=2)
        else:  # day
            time_format = '%m-%d'
            time_locator = mdates.MonthLocator(interval=1)
        
        # 1. Plot main trading parameters time series
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Main Trading Parameters - {period_desc}', fontsize=16, fontweight='bold')
        
        # Main trading parameters
        main_params = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
        
        # Plot main parameters
        for i, param in enumerate(main_params):
            if param in filtered_df.columns:
                row, col = i // 2, i % 2
                axes[row, col].plot(filtered_df.index, filtered_df[param], linewidth=1.2, alpha=0.8)
                axes[row, col].set_title(f'{param.title()} Over Time', fontweight='bold')
                axes[row, col].set_xlabel('Time')
                axes[row, col].set_ylabel(param)
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter(time_format))
                axes[row, col].xaxis.set_major_locator(time_locator)
                axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = f"{output_dir}/main_trading_params_{time_level}_{specific_period or 'auto'}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot detailed volume chart
        if 'volume' in filtered_df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))
            ax.plot(filtered_df.index, filtered_df['volume'], linewidth=1.2, color='darkblue', alpha=0.8)
            ax.set_title(f'Volume Time Series - {period_desc}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Volume')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
            ax.xaxis.set_major_locator(time_locator)
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            filename = f"{output_dir}/volume_time_series_{time_level}_{specific_period or 'auto'}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Plot label time series
        if 'label' in filtered_df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(16, 6))
            ax.plot(filtered_df.index, filtered_df['label'], linewidth=1.2, color='red', alpha=0.8)
            ax.set_title(f'Label Time Series - {period_desc}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Label')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
            ax.xaxis.set_major_locator(time_locator)
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            filename = f"{output_dir}/label_time_series_{time_level}_{specific_period or 'auto'}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Plot sample X features
        x_features = [col for col in filtered_df.columns if col.startswith('X')][:10]
        if x_features:
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            fig.suptitle(f'Sample X Features - {period_desc}', fontsize=16, fontweight='bold')
            
            for i, feature in enumerate(x_features):
                row, col = i // 5, i % 5
                axes[row, col].plot(filtered_df.index, filtered_df[feature], linewidth=1.0, alpha=0.8)
                axes[row, col].set_title(f'{feature}', fontweight='bold')
                axes[row, col].set_xlabel('Time')
                axes[row, col].set_ylabel(feature)
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter(time_format))
                axes[row, col].tick_params(axis='x', rotation=45, labelsize=8)
            
            plt.tight_layout()
            filename = f"{output_dir}/sample_x_features_{time_level}_{specific_period or 'auto'}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"\n{time_level.title()}-level time series charts saved to {output_dir}/ directory")
        print(f"Time range: {period_desc}")
        
    except Exception as e:
        print(f"Failed to generate charts: {e}")
        import traceback
        traceback.print_exc()

def plot_time_series(file_path, output_dir=None):
    """Original plotting function, maintains backward compatibility"""
    print("Using original plotting function - recommend using plot_time_series_multi_level for better visualization")
    
    try:
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving charts to: {output_dir}")
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # 1. Plot main trading parameters time series
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Main Trading Parameters Time Series', fontsize=16, fontweight='bold')
        
        # Main trading parameters
        main_params = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        
        # Plot first 4 main parameters
        for i, param in enumerate(main_params[:4]):
            row, col = i // 2, i % 2
            axes[row, col].plot(df.index, df[param], linewidth=0.8, alpha=0.8)
            axes[row, col].set_title(f'{param.title()} Over Time', fontweight='bold')
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel(param)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/main_trading_params.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot detailed volume chart
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.plot(df.index, df['volume'], linewidth=0.8, color='darkblue', alpha=0.8)
        ax.set_title('Volume Time Series', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/volume_time_series.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot label time series
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.plot(df.index, df['label'], linewidth=0.8, color='red', alpha=0.8)
        ax.set_title('Label Time Series', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Label')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/label_time_series.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Plot sample X features (X1-X10)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Sample X Features (X1-X10) Time Series', fontsize=16, fontweight='bold')
        
        for i in range(10):
            row, col = i // 5, i % 5
            feature = f'X{i+1}'
            axes[row, col].plot(df.index, df[feature], linewidth=0.6, alpha=0.8)
            axes[row, col].set_title(f'{feature}', fontweight='bold')
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel(feature)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].tick_params(axis='x', rotation=45, labelsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_x_features.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Plot parameter distribution comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Parameter Distribution and Statistics', fontsize=16, fontweight='bold')
        
        # Main parameter distributions
        main_params_with_label = main_params + ['label']
        
        for i, param in enumerate(main_params_with_label):
            row, col = i // 3, i % 3
            
            # Plot histogram
            axes[row, col].hist(df[param].dropna(), bins=50, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{param} Distribution', fontweight='bold')
            axes[row, col].set_xlabel(param)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add statistical information
            mean_val = df[param].mean()
            std_val = df[param].std()
            axes[row, col].axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.3f}')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/parameter_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Plot correlation heatmap (main parameters + label)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        corr_data = df[main_params_with_label].corr()
        
        im = ax.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set tick labels
        ax.set_xticks(range(len(main_params_with_label)))
        ax.set_yticks(range(len(main_params_with_label)))
        ax.set_xticklabels(main_params_with_label, rotation=45, ha='right')
        ax.set_yticklabels(main_params_with_label)
        
        # Add numerical labels
        for i in range(len(main_params_with_label)):
            for j in range(len(main_params_with_label)):
                text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Correlation Matrix of Main Parameters', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Plot data quality overview
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Missing values statistics
        missing_counts = df.isnull().sum()
        non_zero_missing = missing_counts[missing_counts > 0]
        
        if len(non_zero_missing) > 0:
            axes[0].bar(range(len(non_zero_missing)), non_zero_missing.values)
            axes[0].set_title('Missing Values by Column', fontweight='bold')
            axes[0].set_xlabel('Column Index')
            axes[0].set_ylabel('Missing Count')
            axes[0].set_xticks(range(len(non_zero_missing)))
            axes[0].set_xticklabels(non_zero_missing.index, rotation=45)
        else:
            axes[0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14, fontweight='bold')
            axes[0].set_title('Missing Values by Column', fontweight='bold')
        
        # Data range statistics
        data_stats = df[main_params_with_label].describe()
        data_ranges = data_stats.loc['max'] - data_stats.loc['min']
        
        axes[1].bar(range(len(data_ranges)), data_ranges.values)
        axes[1].set_title('Data Range by Parameter', fontweight='bold')
        axes[1].set_xlabel('Parameter')
        axes[1].set_ylabel('Range (Max - Min)')
        axes[1].set_xticks(range(len(data_ranges)))
        axes[1].set_xticklabels(data_ranges.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_quality_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTime series charts saved to {output_dir}/ directory:")
        print("1. main_trading_params.png - Main trading parameters time series")
        print("2. volume_time_series.png - Volume time series")
        print("3. label_time_series.png - Label time series")
        print("4. sample_x_features.png - Sample X features time series")
        print("5. parameter_distributions.png - Parameter distribution statistics")
        print("6. correlation_matrix.png - Correlation matrix")
        print("7. data_quality_overview.png - Data quality overview")
        
    except Exception as e:
        print(f"Failed to generate charts: {e}")
        import traceback
        traceback.print_exc()

def analyze_time_periods(file_path):
    """Analyze available time periods in data to help users choose appropriate visualization ranges"""
    try:
        df = pd.read_parquet(file_path)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        print("=== Available Time Period Analysis ===")
        print(f"Total data time range: {df.index.min()} to {df.index.max()}")
        print(f"Total data points: {len(df)}")
        
        # Daily statistics
        daily_counts = df.resample('D').size()
        print(f"\nDaily statistics (top 10 days with most data):")
        top_days = daily_counts.nlargest(10)
        for date, count in top_days.items():
            print(f"  {date.strftime('%Y-%m-%d')}: {count} data points")
        
        # Monthly statistics
        monthly_counts = df.resample('M').size()
        print(f"\nMonthly statistics:")
        for date, count in monthly_counts.items():
            print(f"  {date.strftime('%Y-%m')}: {count} data points")
        
        # Yearly statistics
        yearly_counts = df.resample('Y').size()
        print(f"\nYearly statistics:")
        for date, count in yearly_counts.items():
            print(f"  {date.strftime('%Y')}: {count} data points")
        
        # Recommended visualization periods
        print(f"\n=== Recommended Visualization Periods ===")
        best_day = daily_counts.idxmax().strftime('%Y-%m-%d')
        best_month = monthly_counts.idxmax().strftime('%Y-%m') 
        best_year = yearly_counts.idxmax().strftime('%Y')
        
        print(f"Recommended minute-level visualization date: {best_day} (data points: {daily_counts.max()})")
        print(f"Recommended hourly-level visualization month: {best_month} (data points: {monthly_counts.max()})")
        print(f"Recommended daily-level visualization year: {best_year} (data points: {yearly_counts.max()})")
        
        return {
            'best_day': best_day,
            'best_month': best_month, 
            'best_year': best_year
        }
        
    except Exception as e:
        print(f"Failed to analyze time periods: {e}")
        return None

if __name__ == "__main__":
    file_path = "/ssdwork/gaotianrun/findataset/test_converted.parquet"
    
    # Preview data first
    print("=== Data Preview ===")
    preview_parquet(file_path)
    
    # Analyze available time periods
    print("\n=== Time Period Analysis ===")
    time_periods = analyze_time_periods(file_path)
    
    if time_periods:
        # Use recommended time periods for multi-level visualization
        print("\n=== Starting Multi-level Time Series Visualization ===")
        
        # Minute-level visualization (one day)
        print(f"\n1. Plotting minute-level data ({time_periods['best_day']})...")
        plot_time_series_multi_level(file_path, time_level='minute', 
                                    specific_period=time_periods['best_day'])
        
        # Hourly-level visualization (one month)  
        print(f"\n2. Plotting hourly-level data ({time_periods['best_month']})...")
        plot_time_series_multi_level(file_path, time_level='hour',
                                    specific_period=time_periods['best_month'])
        
        # Daily-level visualization (one year)
        print(f"\n3. Plotting daily-level data ({time_periods['best_year']})...")
        plot_time_series_multi_level(file_path, time_level='day',
                                    specific_period=time_periods['best_year'])
        
        print("\n=== All Visualizations Completed ===")
        print("\nUsage:")
        print("1. Run this script directly to view recommended time period visualizations")
        print("2. Call plot_time_series_multi_level(file_path, 'minute', '2023-06-15') to view specific date")
        print("3. Call plot_time_series_multi_level(file_path, 'hour', '2023-06') to view specific month") 
        print("4. Call plot_time_series_multi_level(file_path, 'day', '2023') to view specific year")
    else:
        print("Time period analysis failed, using original plotting method...")
        plot_time_series(file_path)

    