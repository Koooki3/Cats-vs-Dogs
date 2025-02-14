import os
import torch
import torch.nn as nn
from models.models import get_model
from utils.config import Config
from pathlib import Path
from tabulate import tabulate
import json
from collections import OrderedDict
import numpy as np

class ModelChecker:
    def __init__(self):
        self.config = Config()
        self.model_dir = Path(self.config.model_dir)
        self.available_models = self._scan_models()
        
    def _scan_models(self):
        """扫描模型目录下的所有模型文件"""
        models = []
        print(f"\n正在扫描模型目录: {self.model_dir}")
        
        if not self.model_dir.exists():
            print(f"错误: 模型目录不存在 - {self.model_dir}")
            return models
            
        for file in self.model_dir.glob('*.pth'):
            try:
                info = self._get_model_info(file)
                if info:
                    models.append(info)
                    
            except Exception as e:
                print(f"处理模型文件 {file.name} 时出错: {str(e)}")
                
        return sorted(models, key=lambda x: x.get('accuracy', 0), reverse=True)
    
    def _get_model_info(self, model_path):
        """获取单个模型文件的详细信息"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 获取正确的准确率
            accuracy = checkpoint.get('current_acc', checkpoint.get('best_acc', 0))
            if 'best_model_' in model_path.name:
                # 对于最佳模型文件，使用 best_recorded_acc
                accuracy = checkpoint.get('best_recorded_acc', checkpoint.get('best_acc', 0))
            
            # 基本信息
            info = {
                'filename': model_path.name,
                'model_name': checkpoint.get('model_name', 'unknown'),
                'epoch': checkpoint.get('epoch', -1),
                'accuracy': accuracy,
                'best_recorded_acc': checkpoint.get('best_recorded_acc', accuracy),
                'config': checkpoint.get('config', {}),
                'size_mb': model_path.stat().st_size / (1024 * 1024),
                'last_modified': model_path.stat().st_mtime,
                'total_params': 0,
                'trainable_params': 0,
                'layers': {}
            }
            
            # 尝试加载模型结构
            try:
                if info['model_name'] != 'unknown':
                    model = get_model(info['model_name'], pretrained=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # 获取模型参数统计
                    info['total_params'] = sum(p.numel() for p in model.parameters())
                    info['trainable_params'] = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                    info['layers'] = self._analyze_model_structure(model)
            except Exception as e:
                print(f"警告：无法加载模型结构 {model_path.name}: {str(e)}")
            
            return info
            
        except Exception as e:
            print(f"错误：解析模型文件 {model_path.name} 时出错: {str(e)}")
            return {
                'filename': model_path.name,
                'model_name': 'unknown',
                'epoch': -1,
                'accuracy': 0,
                'best_recorded_acc': 0,
                'config': {},
                'size_mb': model_path.stat().st_size / (1024 * 1024),
                'last_modified': model_path.stat().st_mtime,
                'total_params': 0,
                'trainable_params': 0,
                'layers': {}
            }
    
    def _analyze_model_structure(self, model):
        """分析模型结构"""
        layers_info = OrderedDict()
        
        def collect_info(module, prefix=''):
            for name, layer in module.named_children():
                layer_name = f"{prefix}.{name}" if prefix else name
                
                # 获取层的基本信息
                layer_type = layer.__class__.__name__
                num_params = sum(p.numel() for p in layer.parameters())
                
                # 记录层信息
                if layer_type in layers_info:
                    layers_info[layer_type]['count'] += 1
                    layers_info[layer_type]['params'] += num_params
                else:
                    layers_info[layer_type] = {
                        'count': 1,
                        'params': num_params
                    }
                
                # 递归处理子层
                if len(list(layer.children())) > 0:
                    collect_info(layer, layer_name)
        
        collect_info(model)
        return layers_info
    
    def print_model_summary(self):
        """打印模型概要信息"""
        if not self.available_models:
            print("\n未找到任何模型文件!")
            return
            
        print("\n=== 模型概要信息 ===")
        
        # 准备表格数据
        headers = ["文件名", "模型类型", "轮次", "当前准确率", "历史最佳", "参数量", "大小(MB)"]
        rows = []
        
        for model in self.available_models:
            try:
                rows.append([
                    model.get('filename', 'Unknown'),
                    model.get('model_name', 'Unknown'),
                    model.get('epoch', -1) + 1,
                    f"{model.get('accuracy', 0):.2f}%",
                    f"{model.get('best_recorded_acc', 0):.2f}%",
                    f"{model.get('total_params', 0):,}",
                    f"{model.get('size_mb', 0):.1f}"
                ])
            except Exception as e:
                print(f"警告：处理模型信息时出错: {str(e)}")
                continue
        
        if rows:
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        else:
            print("无法显示模型信息：数据处理错误")
    
    def print_detailed_info(self):
        """打印详细模型信息"""
        if not self.available_models:
            return
            
        for idx, model in enumerate(self.available_models, 1):
            print(f"\n=== 模型 {idx} 详细信息 ===")
            print(f"文件名: {model['filename']}")
            print(f"模型类型: {model['model_name']}")
            print(f"训练轮次: {model['epoch'] + 1}")
            print(f"最佳准确率: {model['accuracy']:.2f}%")
            print(f"文件大小: {model['size_mb']:.1f}MB")
            print(f"总参数量: {model['total_params']:,}")
            print(f"可训练参数: {model['trainable_params']:,}")
            
            # 打印配置信息
            if model['config']:
                print("\n训练配置:")
                for key, value in model['config'].items():
                    print(f"  {key}: {value}")
            
            # 打印层结构统计
            if 'layers' in model:
                print("\n层结构统计:")
                layer_info = []
                for layer_type, info in model['layers'].items():
                    layer_info.append([
                        layer_type,
                        info['count'],
                        f"{info['params']:,}"
                    ])
                print(tabulate(
                    layer_info,
                    headers=["层类型", "数量", "参数数量"],
                    tablefmt="grid"
                ))
            
            print("-" * 50)
    
    def export_info(self, output_path='model_analysis.json'):
        """导出模型信息到JSON文件"""
        if not self.available_models:
            return
            
        # 准备导出数据
        export_data = {
            'analysis_time': str(torch.datetime.datetime.now()),
            'models': self.available_models
        }
        
        # 写入JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n模型分析结果已导出到: {output_path}")

def main():
    checker = ModelChecker()
    
    while True:
        print("\n=== 模型检查工具 ===")
        print("1. 显示模型概要信息")
        print("2. 显示详细模型信息")
        print("3. 导出模型分析报告")
        print("4. 退出")
        
        choice = input("\n请选择功能 (1-4): ")
        
        if choice == '1':
            checker.print_model_summary()
        elif choice == '2':
            checker.print_detailed_info()
        elif choice == '3':
            output_path = input("请输入导出文件路径 [model_analysis.json]: ").strip()
            if not output_path:
                output_path = 'model_analysis.json'
            checker.export_info(output_path)
        elif choice == '4':
            print("\n感谢使用!")
            break
        else:
            print("\n无效的选择，请重试")

if __name__ == '__main__':
    main()
