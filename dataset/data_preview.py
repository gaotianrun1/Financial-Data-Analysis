import pandas as pd
import sys

def preview_parquet(file_path):
    """简单查看parquet文件格式"""
    try:
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        
        print(f"文件: {file_path}")
        print(f"形状: {df.shape} (行数, 列数)")
        print(f"索引类型: {type(df.index).__name__}")
        
        print("\n列名:")
        print(df.columns.tolist())
        
        print("\n数据类型:")
        print(df.dtypes)
        
        print("\n前5行数据:")
        print(df.head())
        
        print("\n后5行数据:")
        print(df.tail())
        
        # print("\n基本统计:")
        # print(df.describe())
        
    except Exception as e:
        print(f"读取文件失败: {e}")

if __name__ == "__main__":
    file_path = "/ssdwork/gaotianrun/findataset/train.parquet"
    preview_parquet(file_path)