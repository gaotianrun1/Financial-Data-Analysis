#!/usr/bin/env python3
"""
Checkpoint功能使用示例

演示如何：
1. 查看已保存的checkpoint
2. 从checkpoint加载模型
3. 从checkpoint恢复训练
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train.checkpoint_utils import (
    print_checkpoint_summary, 
    find_best_checkpoint, 
    find_latest_checkpoint,
    load_checkpoint,
    resume_training_from_checkpoint
)
from model.lstm_model import LSTMModel
from config import CONFIG

def main():
    print("=== Checkpoint功能示例 ===")
    
    # 示例checkpoint目录（请根据实际情况修改）
    checkpoint_dir = "outputs/20241212_143052/checkpoints"  # 替换为实际的checkpoint目录
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint目录不存在: {checkpoint_dir}")
        print("请先运行训练脚本生成checkpoint，或修改checkpoint_dir变量")
        return
    
    # 1. 查看所有checkpoint
    print("\n1. 查看所有可用的checkpoint:")
    print_checkpoint_summary(checkpoint_dir)
    
    # 2. 查找最佳checkpoint
    print("\n2. 查找最佳checkpoint:")
    best_checkpoint = find_best_checkpoint(checkpoint_dir)
    if best_checkpoint:
        print(f"找到最佳checkpoint: {best_checkpoint}")
    else:
        print("未找到最佳checkpoint")
    
    # 3. 查找最新checkpoint
    print("\n3. 查找最新checkpoint:")
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"找到最新checkpoint: {latest_checkpoint}")
    else:
        print("未找到最新checkpoint")
    
    # 4. 从checkpoint加载模型（仅用于推理）
    if best_checkpoint:
        print("\n4. 从最佳checkpoint加载模型:")
        
        # 创建模型实例
        model = LSTMModel(
            input_size=CONFIG["model"]["input_size"],
            hidden_layer_size=CONFIG["model"]["lstm_size"],
            num_layers=CONFIG["model"]["num_lstm_layers"],
            output_size=1,
            dropout=CONFIG["model"]["dropout"]
        )
        
        # 加载checkpoint
        try:
            load_info = load_checkpoint(best_checkpoint, model, device='cpu')
            print(f"模型已从第 {load_info['epoch']} 个epoch的checkpoint加载")
            print(f"该checkpoint的验证损失: {load_info['val_loss']:.6f}")
        except Exception as e:
            print(f"加载checkpoint失败: {e}")
    
    # 5. 演示如何恢复训练（需要优化器和调度器）
    if latest_checkpoint:
        print("\n5. 演示恢复训练的准备:")
        print(f"如果要恢复训练，可以从: {latest_checkpoint}")
        print("需要创建相同的模型、优化器和调度器，然后调用 resume_training_from_checkpoint()")
        
        # 示例代码（不实际执行训练）
        print("\n示例代码:")
        print("""
        from train.checkpoint_utils import resume_training_from_checkpoint
        
        # 创建模型、优化器、调度器
        model = LSTMModel(...)
        optimizer = torch.optim.Adam(model.parameters(), ...)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, ...)
        
        # 从checkpoint恢复
        start_epoch = resume_training_from_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        
        # 继续训练
        for epoch in range(start_epoch, total_epochs):
            # 训练代码...
        """)

if __name__ == "__main__":
    main() 