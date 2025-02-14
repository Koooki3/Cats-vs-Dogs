import logging
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class TrainLogger:
    def __init__(self, config):
        self.config = config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 确保日志目录存在
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.tensorboard_dir, exist_ok=True)
        
        # 设置文件日志
        log_file = os.path.join(
            config.log_dir, 
            f'{config.model_name}_{timestamp}.log'
        )
        
        # 配置logger
        self._setup_logger(log_file)
        
        # 设置tensorboard
        self.tensorboard_dir = os.path.join(
            config.tensorboard_dir,
            f"{config.model_name}_{timestamp}"
        )
        try:
            self.writer = SummaryWriter(self.tensorboard_dir, flush_secs=10)
            self.logger.info(f"TensorBoard事件文件将保存到: {self.tensorboard_dir}")
        except Exception as e:
            self.logger.error(f"TensorBoard初始化失败: {str(e)}")
            self.writer = None
    
    def _setup_logger(self, log_file):
        """设置日志记录器"""
        self.logger = logging.getLogger(f'CatDog_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 添加处理器
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.info(f"日志将保存到: {log_file}")
    
    def log_config(self, config_dict):
        """记录配置信息"""
        self.logger.info("\n=== 训练配置 ===")
        if self.writer:
            config_text = ""
            for key, value in config_dict.items():
                config_text += f"{key}: {value}\n"
                self.logger.info(f"{key}: {value}")
            self.writer.add_text('Configuration', config_text)
            self.writer.flush()
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """记录每个epoch的信息"""
        # 记录到日志文件
        self.logger.info(
            f"\nEpoch {epoch} "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}% "
            f"LR: {lr:.6f}"
        )
        
        # 记录到tensorboard
        if self.writer:
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
            self.writer.flush()  # 立即写入磁盘
    
    def log_step(self, step, loss, accuracy, learning_rate):
        """记录训练步骤信息"""
        if self.writer:
            self.writer.add_scalar('Step/Loss', loss, step)
            self.writer.add_scalar('Step/Accuracy', accuracy, step)
            self.writer.add_scalar('Step/Learning_Rate', learning_rate, step)
            self.writer.flush()
    
    def log_images(self, tag, images, step):
        """记录图像"""
        if self.writer:
            self.writer.add_image(tag, images, step)
            self.writer.flush()
    
    def log_info(self, message):
        """记录信息"""
        self.logger.info(message)
        
    def log_error(self, message):
        """记录错误"""
        self.logger.error(message)
    
    def close(self):
        """关闭日志记录器"""
        if hasattr(self, 'writer') and self.writer:
            self.writer.flush()  # 确保所有数据都写入磁盘
            self.writer.close()
        # 关闭所有日志处理器
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
