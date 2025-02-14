import torch
import os
import multiprocessing

class Config:
    # 基础目录配置
    dataset_dir = './train'
    model_dir = './model/'
    log_dir = './logs/'          # 添加日志目录
    tensorboard_dir = './runs/'  # 添加TensorBoard目录
    pretrained_dir = './Pre-trained models/'  # 添加预训练模型目录
    model_name = None  # 添加模型名称属性
    
    # 训练配置
    batch_size = 16          # 降低默认batch size
    num_epochs = 100
    learning_rate = 3e-4
    weight_decay = 1e-2
    label_smoothing = 0.1
    min_lr = 1e-6
    
    # 数据加载配置
    num_workers = 2          # 降低worker数量
    pin_memory = True
    persistent_workers = True  # 保持worker进程
    prefetch_factor = 2
    
    # 显存优化
    gradient_accumulation_steps = 4  # 梯度累积
    use_amp = True  # 默认开启混合精度训练
    empty_cache_freq = 20           # 频繁清理缓存
    max_grad_norm = 1.0            # 梯度裁剪
    
    # 数据增强配置
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    
    # 训练优化配置
    warmup_epochs = 5
    warmup_lr_init = 1e-6
    
    def __init__(self):
        try:
            from rich.console import Console
            console = Console()
            console.print("\n[bold green]配置初始化:[/bold green]")
            
            # 创建所有必要的目录
            for dir_path in [self.model_dir, self.log_dir, 
                            self.tensorboard_dir, self.pretrained_dir]:  # 添加预训练目录
                os.makedirs(dir_path, exist_ok=True)
                console.print(f"[blue]✓[/blue] 目录已就绪: {dir_path}")
            
            # GPU相关配置
            self.use_cuda = torch.cuda.is_available()
            if self.use_cuda:
                self.device = torch.device('cuda')
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                console.print(f"\n[bold yellow]GPU 信息:[/bold yellow]")
                console.print(f"设备: [cyan]{torch.cuda.get_device_name(0)}[/cyan]")
                console.print(f"显存: [cyan]{gpu_mem:.1f}GB[/cyan]")
                
                # 更保守的显存管理策略
                if gpu_mem < 4:  # 4GB以下显存
                    console.print("[red]警告: 显存较小，已自动调整配置[/red]")
                    self.batch_size = 4
                    self.gradient_accumulation_steps = 8
                    self.num_workers = 1
                    self.prefetch_factor = 1
                elif gpu_mem < 8:  # 8GB以下显存
                    self.batch_size = 8
                    self.gradient_accumulation_steps = 4
                    self.num_workers = 2
                    self.prefetch_factor = 2
                else:  # 8GB及以上显存
                    self.batch_size = 16
                    self.gradient_accumulation_steps = 2
                    self.pin_memory = True
                    self.prefetch_factor = 4
            else:
                self.device = torch.device('cpu')
                self.batch_size = 8
                self.num_workers = 0
                self.pin_memory = False
                console.print("\n[yellow]警告: 未检测到GPU，将使用CPU[/yellow]")
                
        except ImportError:
            # 如果没有rich库，使用普通打印
            # 创建所有必要的目录
            for dir_path in [self.model_dir, self.log_dir, 
                            self.tensorboard_dir, self.pretrained_dir]:  # 添加预训练目录
                os.makedirs(dir_path, exist_ok=True)
            
            # GPU相关配置
            self.use_cuda = torch.cuda.is_available()
            if self.use_cuda:
                self.device = torch.device('cuda')
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # 更保守的显存管理策略
                if gpu_mem < 4:  # 4GB以下显存
                    self.batch_size = 4
                    self.gradient_accumulation_steps = 8
                    self.num_workers = 1
                    self.prefetch_factor = 1
                elif gpu_mem < 8:  # 8GB以下显存
                    self.batch_size = 8
                    self.gradient_accumulation_steps = 4
                    self.num_workers = 2
                    self.prefetch_factor = 2
                else:  # 8GB及以上显存
                    self.batch_size = 16
                    self.gradient_accumulation_steps = 2
                    self.pin_memory = True
                    self.prefetch_factor = 4
            else:
                self.device = torch.device('cpu')
                self.batch_size = 8
                self.num_workers = 0
                self.pin_memory = False
