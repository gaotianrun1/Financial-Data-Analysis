import pandas as pd
import os

def convert_test_index(input_path, output_path, new_start_str):
    # 1. 读原始文件
    df = pd.read_parquet(input_path)
    
    # 2. 构建一个按分钟递增的新索引
    new_start = pd.to_datetime(new_start_str)
    # periods=len(df)：总行数；freq='T' 表示分钟频率
    new_index = pd.date_range(start=new_start, periods=len(df), freq='T')
    
    # 3. 直接替换 DataFrame 的 index
    df.index = new_index
    
    # 4. 保存到新文件（原文件不会被修改）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)
    print(f"已生成：{output_path}")

if __name__ == "__main__":
    inp = "/ssdwork/gaotianrun/findataset/test.parquet"
    outp = "/ssdwork/gaotianrun/findataset/test_converted.parquet"
    # 你想设定的新起始时间：
    new_start = "2024-03-01 00:00:00"
    convert_test_index(inp, outp, new_start)

    df2 = pd.read_parquet("/ssdwork/gaotianrun/findataset/test_converted.parquet")
    print(df2.index[:5])