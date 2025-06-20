import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np
import os
import glob
from datetime import datetime
import pickle

def run_epoch(model, dataloader, criterion, optimizer, config, is_training=False):
    """执行单个epoch的训练或验证"""
    epoch_loss = 0
    num_batches = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]
        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        # 前向传播
        if is_training:
            out = model(x)
        else:
            with torch.no_grad():
                out = model(x)
        
        loss = criterion(out.contiguous(), y.contiguous())

        # 反向传播
        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.detach().item() # TODO: 需要检查这里的loss是每个样本的均值还是某种总值
        num_batches += 1

    # 返回平均损失
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def create_scheduler(optimizer, config):
    """根据配置创建学习率调度器"""
    scheduler_type = config["training"].get("scheduler_type", "step")
    num_epochs = config["training"]["num_epoch"]
    
    if scheduler_type == "cosine":
        # 余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            eta_min=config["training"]["learning_rate"] * 0.01  # 最小学习率为初始学习率的1%
        )
    else:
        # 默认使用StepLR
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config["training"]["scheduler_step_size"], 
            gamma=0.1
        )
    
    return scheduler

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   checkpoint_dir, filename_prefix="checkpoint"):
    """
    保存模型checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        train_loss: 训练损失
        val_loss: 验证损失
        checkpoint_dir: checkpoint保存目录
        filename_prefix: 文件名前缀
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 构建checkpoint数据
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # 确定文件名
    filename = f"{filename_prefix}_epoch_{epoch:04d}.pth"
    filepath = os.path.join(checkpoint_dir, filename)
    
    # 保存checkpoint
    torch.save(checkpoint, filepath)
    
    return filepath

def manage_checkpoints(checkpoint_dir, filename_prefix="checkpoint", max_keep=5):
    """
    管理checkpoint文件，删除多余的checkpoint
    
    Args:
        checkpoint_dir: checkpoint目录
        filename_prefix: 文件名前缀
        max_keep: 最多保留的checkpoint数量
    """
    if max_keep <= 0:
        return  # 保留所有checkpoint
    
    # 找到所有非best的checkpoint文件
    pattern = os.path.join(checkpoint_dir, f"{filename_prefix}_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if len(checkpoint_files) <= max_keep:
        return  # 数量未超过限制
    
    # 按照epoch号排序
    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    
    # 删除最旧的checkpoint
    files_to_delete = checkpoint_files[:-max_keep]
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"删除旧checkpoint: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"删除checkpoint失败 {file_path}: {e}")

def train_model(model, dataset_train, dataset_val, config, checkpoint_dir=None):
    """
    训练模型，包含checkpoint保存和训练历史记录功能
    
    Args:
        model: 模型
        dataset_train: 训练数据集
        dataset_val: 验证数据集
        config: 配置字典
        checkpoint_dir: checkpoint保存目录
    
    Returns:
        model: 训练后的模型
        training_history: 训练历史字典，包含完整的损失曲线等信息
    """
    # 创建DataLoader
    train_dataloader = DataLoader(
        dataset_train, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset_val, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False  # 验证时不需要shuffle
    )

    # 创建损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        # betas=(0.9, 0.98), 
        # eps=1e-9
    )
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config)

    # 初始化checkpoint设置
    checkpoint_config = config["training"].get("checkpoint", {})
    checkpoint_enabled = checkpoint_config.get("enabled", False)
    checkpoint_interval = checkpoint_config.get("save_interval", 500)
    max_keep = checkpoint_config.get("max_keep", 5)
    
    if checkpoint_enabled and checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化训练历史 - 包含完整信息用于分析和可视化
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'epochs': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0,
        'total_epochs': 0,
        'checkpoints_saved': [], 
    }

    for epoch in range(config["training"]["num_epoch"]):
        train_loss = run_epoch(
            model, train_dataloader, criterion, optimizer, config, is_training=True
        )
        
        val_loss = run_epoch(
            model, val_dataloader, criterion, optimizer, config, is_training=False
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存训练历史
        if config["training"]["save_history"]:
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['learning_rate'].append(current_lr)
            training_history['epochs'].append(epoch + 1)

            if val_loss < training_history['best_val_loss']:
                training_history['best_val_loss'] = val_loss
                training_history['best_epoch'] = epoch + 1
        
        # 定期保存checkpoint和training history
        if (checkpoint_enabled and checkpoint_dir and 
            (epoch + 1) % checkpoint_interval == 0):
            try:
                checkpoint_path = save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, 
                    train_loss, val_loss, checkpoint_dir
                )
                print(f"保存checkpoint: {os.path.basename(checkpoint_path)}")
                training_history['checkpoints_saved'].append({
                    'epoch': epoch + 1,
                    'type': 'interval',
                    'path': checkpoint_path,
                    'val_loss': val_loss
                })
                
                # 管理checkpoint数量
                manage_checkpoints(checkpoint_dir, max_keep=max_keep)

                # 保存训练历史
                history_filename = f"training_history_epoch_{epoch+1:04d}.pth"
                history_filepath = os.path.join(checkpoint_dir, history_filename)
                with open(history_filepath, 'wb') as f:
                    pickle.dump(training_history, f)
                print(f"训练历史已保存到: {history_filename}")
                
            except Exception as e:
                print(f"保存checkpoint失败: {e}")
        
        # 打印训练进度
        print_interval = config["training"]["print_interval"]
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f'Epoch[{epoch + 1:3d}/{config["training"]["num_epoch"]:3d}] | '
                  f'Loss: train={train_loss:.6f}, val={val_loss:.6f} | '
                  f'LR={current_lr:.2e} | '
                  f'Best: {training_history["best_val_loss"]:.6f}@{training_history["best_epoch"]}')
    
    training_history['total_epochs'] = config["training"]["num_epoch"]
    
    print(f'\n=== 训练完成 ===')
    print(f'实际训练轮数: {training_history["total_epochs"]}')
    print(f'最佳验证损失: {training_history["best_val_loss"]:.6f} (第{training_history["best_epoch"]}轮)')

    return model, training_history 