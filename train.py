import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.nn as nn
import torchvision
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import gc
import psutil
from tqdm import tqdm
import time
import argparse
from pathlib import Path

from data.dataset import CatDogDataset
from models.models import get_model
from utils.config import Config
from utils.logger import TrainLogger
from timm.data.mixup import Mixup
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich import box

console = Console()

class Trainer:
    def __init__(self, args, config):
        # 确保数据目录存在并包含数据
        if not os.path.exists(config.dataset_dir):
            raise ValueError(f"数据目录不存在: {config.dataset_dir}")
        
        if len(os.listdir(config.dataset_dir)) == 0:
            raise ValueError(f"数据目录为空: {config.dataset_dir}")
        
        self.args = args
        self.config = config
        # 将模型名称添加到config中
        self.config.model_name = args.model_name
        self.logger = TrainLogger(config)
        
        # 设置设备
        self.device = config.device
        
        # 加载数据
        self._setup_dataloaders()
        
        # 初始化模型
        self._setup_model()
        
        # 设置训练组件
        self._setup_training_components()
        
        # 加载检查点（如果存在）
        self.start_epoch = self._load_checkpoint() if args.resume else 0
        
        self.best_acc = 0
        self.best_loss = float('inf')
        self._adjust_batch_size()
        print(f"当前batch size: {self.config.batch_size}")
        self.console = Console()
    
    def _adjust_batch_size(self):
        """根据可用显存动态调整batch size"""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_mem_gb = gpu_mem / (1024**3)
            
            # 根据显存大小调整batch size和梯度累积
            if gpu_mem_gb < 4:  # 小于4GB显存
                self.config.batch_size = max(4, self.config.batch_size // 4)
                self.config.gradient_accumulation_steps *= 2
            elif gpu_mem_gb < 8:  # 小于8GB显存
                self.config.batch_size = max(8, self.config.batch_size // 2)
                self.config.gradient_accumulation_steps *= 2
    
    def _setup_dataloaders(self):
        """设置数据加载器"""
        train_dataset = CatDogDataset(self.config.dataset_dir, mode='train')
        val_dataset = CatDogDataset(self.config.dataset_dir, mode='val')
        
        # 确保drop_last=True，防止最后一个batch只有一个样本
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True  # 添加这行
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            drop_last=False  # 验证时可以保留最后一个batch
        )
    
    def _setup_model(self):
        """设置模型"""
        self.model = get_model(
            self.args.model_name,
            pretrained=self.args.pretrained
        ).to(self.device)
        
        # 可视化模型结构
        if hasattr(self.logger, 'writer'):
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            self.logger.writer.add_graph(self.model, dummy_input)
    
    def _setup_training_components(self):
        """设置训练组件"""
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.num_epochs // 3,
            eta_min=self.config.min_lr
        )
        self.scaler = GradScaler() if self.config.use_amp else None
        
        self.mixup_fn = Mixup(
            mixup_alpha=self.config.mixup_alpha,
            cutmix_alpha=self.config.cutmix_alpha,
            num_classes=2
        )
    
    def _save_checkpoint(self, epoch, is_best=False, val_acc=None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_name': self.args.model_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': val_acc,  # 修改这里：保存实际的验证准确率
            'current_acc': val_acc,  # 添加这个字段来区分当前准确率
            'best_recorded_acc': self.best_acc,  # 添加这个字段来记录历史最佳准确率
            'best_loss': self.best_loss,
            'config': {
                'image_size': 224,
                'num_classes': 2,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size
            }
        }
        
        try:
            # 根据是否是最佳模型选择保存路径和准确率
            if is_best:
                model_path = os.path.join(
                    self.config.model_dir, 
                    f'best_model_{self.args.model_name}.pth'
                )
                checkpoint['best_acc'] = self.best_acc  # 对于最佳模型，使用历史最佳准确率
                save_msg = f"保存最佳模型 (准确率: {self.best_acc:.2f}%)"
            else:
                model_path = os.path.join(self.config.model_dir, 'last_model.pth')
                save_msg = f"保存最新模型 (准确率: {val_acc:.2f}%)"
            
            torch.save(checkpoint, model_path)
            print(f"{save_msg}: {model_path}")
                
        except Exception as e:
            print(f"保存检查点时出错: {str(e)}")

    def _load_checkpoint(self):
        """加载检查点"""
        # 首先尝试加载通用的最新模型
        last_model_path = os.path.join(self.config.model_dir, 'last_model.pth')
        if not os.path.exists(last_model_path):
            print("未找到之前的检查点，将从头开始训练")
            return 0
        
        print(f"加载最新模型: {last_model_path}")
        try:
            # 使用weights_only=False以加载完整检查点
            checkpoint = torch.load(
                last_model_path,
                map_location=self.device,
                weights_only=False  # 修改这里以加载完整检查点
            )
            
            # 验证模型名称匹配
            if checkpoint.get('model_name') != self.args.model_name:
                print("最新模型与当前选择的模型不匹配，将从头开始训练")
                return 0
            
            # 加载模型权重    
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载训练状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复最佳指标
            self.best_acc = checkpoint.get('best_acc', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            
            # 获取已训练的轮数
            epoch = checkpoint.get('epoch', -1)
            
            # 打印恢复状态
            print(f"成功恢复训练状态:")
            print(f"- 已训练轮数: {epoch + 1}")
            print(f"- 最佳准确率: {self.best_acc:.2f}%")
            print(f"- 学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 验证训练状态
            if hasattr(self.model, 'training'):
                self.model.train()
                
            # 确保优化器状态正确
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            
            return epoch + 1
                
        except Exception as e:
            print(f"加载检查点时出错: {str(e)}")
            print("将从头开始训练")
            return 0
    
    @torch.no_grad()
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return val_loss / len(self.val_loader), 100. * correct / total
    
    def train(self):
        """训练模型"""
        total_epochs = self.args.epochs
        
        # 记录初始配置
        self.logger.log_config({
            'model_name': self.args.model_name,
            'epochs': total_epochs,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'device': str(self.device)
        })
        
        # 使用rich创建训练信息面板
        console.print(Panel(
            f"[bold green]开始训练[/bold green]\n"
            f"模型: [cyan]{self.args.model_name.upper()}[/cyan]\n"
            f"轮次: [yellow]{self.start_epoch}[/yellow]/[yellow]{total_epochs}[/yellow]\n"
            f"Batch Size: [magenta]{self.config.batch_size}[/magenta]\n"
            f"学习率: [blue]{self.config.learning_rate}[/blue]\n"
            f"设备: [red]{self.device}[/red]",
            title="训练配置",
            border_style="green"
        ))

        for epoch in range(self.start_epoch, total_epochs):
            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 使用 global_step 来跟踪总训练步数
            global_step = epoch * len(self.train_loader)
            
            # 验证后记录到tensorboard
            self.logger.log_epoch(
                epoch=epoch,  # 这里改为从0开始计数
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=current_lr
            )
            
            # 创建训练结果表格
            table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            table.add_column("指标", style="cyan")
            table.add_column("训练", style="green")
            table.add_column("验证", style="yellow")
            
            table.add_row(
                "Loss",
                f"{train_loss:.4f}",
                f"{val_loss:.4f}"
            )
            table.add_row(
                "Accuracy",
                f"{train_acc:.2f}%",
                f"{val_acc:.2f}%"
            )
            table.add_row(
                "Learning Rate",
                f"{self.optimizer.param_groups[0]['lr']:.6f}",
                ""
            )
            
            console.print(f"\n[bold]Epoch {epoch+1}/{total_epochs} 结果:[/bold]")
            console.print(table)
            
            # 先保存最新模型（使用当前验证准确率）
            self._save_checkpoint(epoch, is_best=False, val_acc=val_acc)
            self.logger.log_info(f"保存最新模型: epoch {epoch+1}, 准确率: {val_acc:.2f}%")
            
            # 如果是最佳模型则额外保存（使用最佳准确率）
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint(epoch, is_best=True, val_acc=val_acc)
                console.print(f"[bold green]新的最佳模型! 准确率: {val_acc:.2f}%[/bold green]")
                self.logger.log_info(
                    f"新的最佳模型! 准确率: {val_acc:.2f}%"
                )
        
        # 记录训练完成信息
        self.logger.log_info(
            f"训练完成! 最佳准确率: {self.best_acc:.2f}%\n"
            f"模型保存目录: {self.config.model_dir}"
        )
        self.logger.close()
        
        # 训练结束显示
        console.print(Panel(
            f"[bold green]训练完成![/bold green]\n"
            f"最佳准确率: [yellow]{self.best_acc:.2f}%[/yellow]\n"
            f"模型已保存至: [blue]{self.config.model_dir}[/blue]",
            title="训练结束",
            border_style="green"
        ))
        
        self.logger.close()
    
    def _train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        try:
            # 使用tqdm替代rich的Progress
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}",
                ncols=100,
                unit="batch",
                leave=True
            )
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # 计算全局步数
                global_step = epoch * len(self.train_loader) + batch_idx
                
                # 定期清理缓存
                if batch_idx % self.config.empty_cache_freq == 0:
                    torch.cuda.empty_cache()
                
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # 使用混合精度训练
                with autocast('cuda', enabled=self.config.use_amp):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) / self.config.gradient_accumulation_steps
                
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                
                # 统计（使用原始loss，不除以累积步数）
                train_loss += loss.item() * self.config.gradient_accumulation_steps
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                avg_loss = train_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'Acc': f"{accuracy:.2f}%"
                })
                
                # 每N步记录一次训练状态
                if batch_idx % 10 == 0:  # 可以调整记录频率
                    self.logger.log_step(
                        step=global_step,
                        loss=avg_loss,
                        accuracy=accuracy,
                        learning_rate=self.optimizer.param_groups[0]['lr']
                    )
                
                # 定期记录训练状态到tensorboard
                if batch_idx % 10 == 0:  # 每10个批次记录一次
                    step = epoch * len(self.train_loader) + batch_idx
                    self.logger.log_step(
                        step=step,
                        loss=avg_loss,
                        accuracy=accuracy,
                        learning_rate=self.optimizer.param_groups[0]['lr']
                    )
                
                # 记录示例图片
                if hasattr(self.logger, 'writer') and batch_idx % 100 == 0:
                    grid = torchvision.utils.make_grid(inputs[:8].cpu())
                    self.logger.log_images('train_samples', grid, global_step)
            
            # 确保至少有一个样本被处理
            if total == 0:
                raise RuntimeError("没有处理任何样本")
                
            # 计算平均损失和准确率
            avg_loss = train_loss / len(self.train_loader)
            accuracy = 100. * correct / total
            
            return avg_loss, accuracy
            
        except Exception as e:
            console.print(f"[bold red]训练过程出错:[/bold red] {str(e)}")
            self.logger.log_error(f"训练过程出错: {str(e)}")
            # 清理资源
            torch.cuda.empty_cache()
            if hasattr(self, 'train_loader'):
                self.train_loader._iterator = None
            raise e  # 重新抛出异常以便调试

def get_gpu_suggestion():
    """根据GPU情况给出batch_size建议"""
    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_mem >= 8:  # RTX 2050及以上
            return "建议batch_size范围: 32-64"
        elif gpu_mem >= 4:
            return "建议batch_size范围: 16-32"
        else:
            return "建议batch_size范围: 8-16"
    except:
        return "未检测到GPU，建议batch_size: 8"

# 修改模型选项
def get_user_config():
    """获取用户训练配置"""
    print("\n=== 猫狗分类模型训练配置 ===")
    
    # 更新可用模型列表
    available_models = ['seresnet', 'lenet', 'resnet', 'alexnet', 'squeezenet']
    print("\n可用模型:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    while True:
        try:
            model_idx = int(input("\n请选择模型 (输入序号): ")) - 1
            if 0 <= model_idx < len(available_models):
                model_name = available_models[model_idx]
                break
            print("无效的选择，请重试")
        except ValueError:
            print("请输入有效的数字")
    
    # 2. 训练轮数
    while True:
        try:
            epochs = int(input("\n请输入训练轮数 (建议50-200): "))
            if epochs > 0:
                break
            print("轮数必须大于0")
        except ValueError:
            print("请输入有效的数字")
    
    # 3. 批次大小
    gpu_suggestion = get_gpu_suggestion()
    print(f"\n{gpu_suggestion}")
    while True:
        try:
            batch_size = int(input("请输入batch size: "))
            if batch_size > 1:  # 修改这里，确保batch size至少为2
                break
            print("batch size必须大于1")
        except ValueError:
            print("请输入有效的数字")
    
    # 4. 是否使用预训练模型
    while True:
        pretrained = input("\n是否使用预训练模型? (y/n): ").lower()
        if pretrained in ['y', 'n']:
            pretrained = (pretrained == 'y')
            break
        print("请输入 y 或 n")
    
    # 5. 是否开启混合精度训练
    while True:
        amp = input("\n是否开启混合精度训练 (AMP)? (y/n) [y]: ").lower()
        if amp == '':
            amp = 'y'
        if amp in ['y', 'n']:
            amp = (amp == 'y')
            break
        print("请输入 y 或 n")
    
    # 6. 是否继续上次的训练 
    while True:
        resume = input("\n是否继续上次的训练? (y/n): ").lower()
        if resume in ['y', 'n']:
            resume = (resume == 'y')
            break
        print("请输入 y 或 n")
    
    # 确认配置时添加 AMP 信息
    print("\n=== 训练配置确认 ===")
    print(f"选择的模型: {model_name}")
    print(f"训练轮数: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"使用预训练: {'是' if pretrained else '否'}")
    print(f"混合精度训练: {'是' if amp else '否'}")
    print(f"继续训练: {'是' if resume else '否'}")
    
    while True:
        confirm = input("\n确认开始训练? (y/n): ").lower()
        if confirm == 'n':
            print("已取消训练")
            exit()
        elif confirm == 'y':
            break
        print("请输入 y 或 n")
    
    # 创建配置对象
    config = Config()
    
    # 创建参数对象
    args = type('Args', (), {
        'model_name': model_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'pretrained': pretrained,
        'resume': resume,
        'data_dir': './train'
    })
    
    # 更新配置
    config.batch_size = batch_size
    config.num_epochs = epochs
    config.use_amp = amp
    
    return args, config

def main():
    # 获取用户配置
    args, config = get_user_config()
    
    # 创建训练器并开始训练
    trainer = Trainer(args, config)
    trainer.train()

if __name__ == '__main__':
    main()
