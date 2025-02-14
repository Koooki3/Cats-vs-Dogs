import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import os
from PIL import Image, ImageFile
import logging
import numpy as np
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CatDogDataset(Dataset):
    """猫狗分类数据集类"""
    
    # 类变量，用于存储数据集信息和控制初始化
    _dataset_initialized = False
    _train_images = None
    _train_labels = None
    _val_images = None
    _val_labels = None
    
    # 图像参数
    IMAGE_SIZE = 224
    MIN_SIZE = 32
    MAX_SIZE = 1000
    TRAIN_SPLIT = 0.8
    
    # 数据增强和预处理
    base_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 修改数据增强策略，移除有问题的 AutoAugment
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    def __init__(self, data_dir, mode='train', transform=None, seed=42):
        """
        初始化数据集
        Args:
            data_dir (str): 数据目录路径
            mode (str): 'train' 或 'val'
            transform: 自定义的数据转换
            seed (int): 随机种子，确保划分可重复
        """
        self.data_dir = data_dir
        self.mode = mode
        
        # 设置转换器
        self.transform = transform if transform else (
            self.train_transform if mode == 'train' else self.base_transform
        )
        
        # 简化日志，只在首次初始化时打印基本信息
        if not CatDogDataset._dataset_initialized:
            print(f"正在加载数据集...")
            self._initialize_dataset(seed)
            print(f"数据集加载完成!")
        
        # 根据mode选择对应的数据
        self.images = (CatDogDataset._train_images if mode == 'train' 
                      else CatDogDataset._val_images)
        self.labels = (CatDogDataset._train_labels if mode == 'train' 
                      else CatDogDataset._val_labels)

    def _initialize_dataset(self, seed):
        """初始化数据集（仅在第一次调用时执行）"""
        # 加载所有图片
        all_images = []
        all_labels = []
        
        for file in os.listdir(self.data_dir):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            filepath = os.path.join(self.data_dir, file)
            try:
                with Image.open(filepath) as img:
                    img.verify()
                with Image.open(filepath) as img:
                    img.convert('RGB')
                
                label = 1 if '_dog.' in file.lower() else 0
                all_images.append(filepath)
                all_labels.append(label)
                
            except Exception as e:
                CatDogDataset._logger.warning(f"跳过损坏的图片 {filepath}: {str(e)}")
        
        # 划分数据集
        train_images, val_images, train_labels, val_labels = train_test_split(
            all_images, 
            all_labels,
            train_size=self.TRAIN_SPLIT,
            random_state=seed,
            stratify=all_labels
        )
        
        # 存储到类变量
        CatDogDataset._train_images = train_images
        CatDogDataset._train_labels = train_labels
        CatDogDataset._val_images = val_images
        CatDogDataset._val_labels = val_labels
        
        # 简化输出信息
        print(f"总数据量: {len(all_images)}")
        print(f"训练集: {len(train_images)} | 验证集: {len(val_images)}")
        
        CatDogDataset._dataset_initialized = True

    def _setup_logger(self):
        """设置日志记录器"""
        if not CatDogDataset._logger.handlers:  # 只在没有处理器时添加
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            CatDogDataset._logger.addHandler(handler)
            CatDogDataset._logger.setLevel(logging.INFO)

    def __len__(self):
        """
        返回数据集大小
        Returns:
            int: 数据集中的样本总数
        """
        return len(self.images) if hasattr(self, 'images') else 0

    def check_and_fix_image(self, img):
        """检查并修复图像尺寸"""
        w, h = img.size
        
        # 检查图像是否太小或太大
        if w < self.MIN_SIZE or h < self.MIN_SIZE:
            return None, "图片尺寸太小"
        if w > self.MAX_SIZE or h > self.MAX_SIZE:
            # 如果图片太大，先进行缩放
            ratio = min(self.MAX_SIZE/w, self.MAX_SIZE/h)
            new_size = (int(w*ratio), int(h*ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 处理非正方形图片
        if w != h:
            # 计算填充值
            size = max(w, h)
            new_img = Image.new('RGB', (size, size), (255, 255, 255))
            # 居中粘贴
            offset = ((size - w) // 2, (size - h) // 2)
            new_img.paste(img, offset)
            img = new_img
        
        return img, None

    def __getitem__(self, idx):
        """获取单个数据样本"""
        try:
            img_path = self.images[idx]
            label = self.labels[idx]
            
            # 最多尝试3次加载和转换
            max_tries = 3
            for attempt in range(max_tries):
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        
                        # 检查和修复图像
                        img, error = self.check_and_fix_image(img)
                        if error:
                            self.logger.error(f"{error}: {img_path}")
                            break
                        
                        # 应用转换
                        img_tensor = self.transform(img)
                        
                        # 验证张量
                        if img_tensor.shape[1:] != (self.IMAGE_SIZE, self.IMAGE_SIZE):
                            raise ValueError(f"张量尺寸错误: {img_tensor.shape}")
                        
                        if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                            raise ValueError("张量包含无效值")
                        
                        return img_tensor, label
                        
                except Exception as e:
                    if attempt == max_tries - 1:  # 最后一次尝试
                        self.logger.error(f"处理失败 {img_path}: {str(e)}")
                        # 返回一个有效的替代样本
                        return self._get_fallback_sample()
            
            # 如果所有尝试都失败
            return self._get_fallback_sample()
            
        except Exception as e:
            self.logger.error(f"严重错误 {img_path if 'img_path' in locals() else 'unknown'}: {str(e)}")
            return self._get_fallback_sample()

    def _get_fallback_sample(self):
        """返回一个有效的替代样本"""
        # 创建一个全黑图片作为替代
        img_tensor = torch.zeros((3, self.IMAGE_SIZE, self.IMAGE_SIZE))
        if self.transform and hasattr(self.transform, 'normalizer'):
            img_tensor = self.transform.normalizer(img_tensor)
        return img_tensor, 0  # 返回空图片和猫的标签作为默认值

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    创建数据加载器
    Args:
        data_dir (str): 数据目录
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
    Returns:
        tuple: (训练集加载器, 验证集加载器)
    """
    # 创建训练集和验证集
    train_dataset = CatDogDataset(data_dir, mode='train')
    val_dataset = CatDogDataset(data_dir, mode='val')
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    # 简化的功能测试
    dataset = CatDogDataset('./train', mode='train')
    print(f"训练集大小: {len(dataset)}")
    
    val_dataset = CatDogDataset('./train', mode='val')
    print(f"验证集大小: {len(val_dataset)}")
    
    img, label = dataset[0]
    print(f"图像形状: {img.shape}")
