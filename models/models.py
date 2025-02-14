import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, alexnet, squeezenet1_1
from torchvision.models import ResNet18_Weights, AlexNet_Weights, SqueezeNet1_1_Weights
from typing import Optional
import os

# 配置预训练模型保存路径
PRETRAINED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Pre-trained models')
os.makedirs(PRETRAINED_PATH, exist_ok=True)
os.environ['TORCH_HOME'] = PRETRAINED_PATH  # 设置PyTorch预训练模型的下载目录

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SEResNet(nn.Module):
    """带有SE注意力机制的ResNet变体"""
    def __init__(self, num_classes=2):
        super(SEResNet, self).__init__()
        
        # 加深网络结构
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # 增加残差层数量
        self.layer1 = self.make_layer(64, 64, 3)    # 2->3
        self.layer2 = self.make_layer(64, 128, 4, stride=2)  # 2->4
        self.layer3 = self.make_layer(128, 256, 6, stride=2)  # 2->6
        self.layer4 = self.make_layer(256, 512, 3, stride=2)  # 新增
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)  # 增加dropout率
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class LeNet(nn.Module):
    """改进的LeNet模型"""
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 调整全连接层以适应224x224输入
        self.classifier = nn.Sequential(
            nn.Linear(16 * 53 * 53, 120),  # 根据224x224输入调整
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ModifiedResNet(nn.Module):
    """修改后的ResNet模型"""
    def __init__(self, num_classes: int = 2, weights=None):
        super(ModifiedResNet, self).__init__()
        print(f"\n预训练模型将下载到: {PRETRAINED_PATH}")
        self.model = resnet18(weights=weights)
        
        # 1. 增强特征提取能力
        self.model.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 2. 添加注意力机制
        self.attention = SEBlock(512)
        
        # 3. 改进分类头，移除可能导致问题的BatchNorm1d
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 提取特征
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        # 应用注意力
        x = self.attention(x)
        
        # 分类
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return F.log_softmax(x, dim=1)

class ModifiedAlexNet(nn.Module):
    """修改后的AlexNet模型"""
    def __init__(self, num_classes: int = 2, weights=None):
        super(ModifiedAlexNet, self).__init__()
        print(f"\n预训练模型将下载到: {PRETRAINED_PATH}")
        self.model = alexnet(weights=weights)
        
        # 1. 优化第一个卷积层以更好地处理高分辨率图像
        self.model.features[0] = nn.Conv2d(3, 64, kernel_size=11, stride=3, padding=4)
        
        # 2. 添加批归一化层
        new_features = []
        for layer in self.model.features:
            new_features.append(layer)
            if isinstance(layer, nn.Conv2d):
                new_features.append(nn.BatchNorm2d(layer.out_channels))
        self.model.features = nn.Sequential(*new_features)
        
        # 3. 改进分类器，移除BatchNorm1d
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

class ModifiedSqueezeNet(nn.Module):
    def __init__(self, num_classes: int = 2, weights=None):
        super(ModifiedSqueezeNet, self).__init__()
        print(f"\n预训练模型将下载到: {PRETRAINED_PATH}")
        self.model = squeezenet1_1(weights=weights)
        
        # 1. 优化特征提取
        # 修改第一层以更好地处理输入
        self.model.features[0] = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        
        # 2. 添加注意力机制
        self.attention = SEBlock(512)
        
        # 3. 改进分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.attention(x)
        x = self.classifier(x)
        return F.log_softmax(x.view(x.size(0), -1), dim=1)

# 修改 get_model 函数
def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    获取指定的模型
    Args:
        model_name: 模型名称 ('seresnet', 'lenet', 'resnet', 'alexnet', 'squeezenet')
        pretrained: 是否使用预训练权重
    Returns:
        nn.Module: 选择的模型
    """
    models = {
        'seresnet': SEResNet(),
        'lenet': LeNet(),
        'resnet': ModifiedResNet(2, weights=ResNet18_Weights.DEFAULT if pretrained else None),
        'alexnet': ModifiedAlexNet(2, weights=AlexNet_Weights.DEFAULT if pretrained else None),
        'squeezenet': ModifiedSqueezeNet(2, weights=SqueezeNet1_1_Weights.DEFAULT if pretrained else None)
    }
    
    if model_name not in models:
        raise ValueError(
            f"不支持的模型: {model_name}. "
            f"可用模型: {list(models.keys())}"
        )
    
    return models[model_name]

if __name__ == '__main__': 
    # 测试代码
    def test_models():
        # 创建示例输入
        x = torch.randn(1, 3, 250, 250)
        
        for model_name in ['lenet', 'resnet', 'alexnet', 'squeezenet']:
            print(f"\n测试 {model_name}:")
            model = get_model(model_name, pretrained=False)
            y = model(x)
            print(f"输出形状: {y.shape}")
            print(f"参数数量: {sum(p.numel() for p in model.parameters())}")
    
    test_models()
