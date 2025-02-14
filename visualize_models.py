import os
import sys
import torch
import argparse
from pathlib import Path
from torchviz import make_dot
import graphviz
from models.models import get_model, SEBlock
from utils.config import Config
from tabulate import tabulate
import json
from collections import OrderedDict
import torch.nn as nn

class ModelVisualizer:
    def __init__(self, verbose=False):
        self.config = Config()
        self.model_dir = Path(self.config.model_dir)
        self.output_dir = Path(self.config.model_dir) / 'model_architectures'
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device('cpu')
        self.verbose = verbose
        self.available_models = self._scan_models()

    def _scan_models(self):
        """扫描所有模型文件"""
        models = []
        for file in self.model_dir.glob('*.pth'):
            try:
                info = self._get_model_info(file)
                if info:
                    models.append(info)
            except Exception as e:
                if self.verbose:
                    print(f"跳过文件 {file.name}: {str(e)}")
        return sorted(models, key=lambda x: x.get('accuracy', 0), reverse=True)

    def generate_architecture(self, model_path, model_name):
        """生成更美观的模型架构图"""
        try:
            model = get_model(model_name, pretrained=False)
            model.eval()

            # 创建有向图
            dot = graphviz.Digraph(
                name=f"{model_name}_architecture",
                comment=f"{model_name.upper()} Architecture",
                engine='dot'
            )

            # 更新图形整体属性，改进布局和美观度
            dot.attr(
                rankdir='TB',          # 从上到下布局
                compound='true',       # 允许跨子图的边
                splines='polyline',    # 使用多段线而不是正交线
                nodesep='1.0',         # 增加水平间距
                ranksep='1.2',         # 增加垂直间距
                fontname='Arial',
                fontsize='16',
                bgcolor='white',
                concentrate='false',    # 禁用边的合并以保持清晰
                overlap='false',        # 防止节点重叠
                sep='+25,25',          # 增加整体间距
                pad='0.5'              # 增加边距
            )

            # 美化配色方案
            colors = {
                'input': '#E3F2FD',    # 淡蓝色
                'conv': '#C8E6C9',     # 淡绿色
                'pool': '#FFCDD2',     # 淡红色
                'se': '#F8BBD0',       # 粉色
                'fc': '#FFE0B2',       # 橙色
                'output': '#E1BEE7',   # 紫色
                'edge': '#546E7A',     # 深灰蓝色
                'title': '#37474F'     # 深色标题
            }

            def get_node_style(layer_type, is_first=False, is_last=False):
                """获取增强的节点样式"""
                base_style = {
                    'style': 'filled,rounded',
                    'fontname': 'Arial',
                    'fontsize': '12',
                    'penwidth': '2',
                    'margin': '0.3'
                }
                
                if is_first:
                    return {**base_style, 
                           'fillcolor': colors['input'],
                           'shape': 'box',
                           'penwidth': '3'}
                elif is_last:
                    return {**base_style, 
                           'fillcolor': colors['output'],
                           'shape': 'box',
                           'penwidth': '3'}
                elif 'Conv' in layer_type:
                    return {**base_style, 
                           'fillcolor': colors['conv'],
                           'shape': 'box'}
                elif 'Pool' in layer_type:
                    return {**base_style, 
                           'fillcolor': colors['pool'],
                           'shape': 'oval'}
                elif 'SE' in layer_type:
                    return {**base_style, 
                           'fillcolor': colors['se'],
                           'shape': 'diamond'}
                else:
                    return {**base_style, 
                           'fillcolor': colors['fc'],
                           'shape': 'box'}

            def get_layer_label(layer, name, is_first=False, is_last=False):
                """生成更详细的层标签"""
                if is_first:
                    return f"Input\n(3, 224, 224)"
                elif is_last:
                    return f"Output\n(2)"
                elif isinstance(layer, nn.Conv2d):
                    return f"Conv2D\n{layer.in_channels}→{layer.out_channels}\n{layer.kernel_size[0]}×{layer.kernel_size[1]}\nstride={layer.stride[0]}"
                elif isinstance(layer, nn.Linear):
                    return f"Linear\n{layer.in_features}→{layer.out_features}"
                elif isinstance(layer, SEBlock):
                    return f"SE Block\nChannel\nAttention"
                elif isinstance(layer, nn.MaxPool2d):
                    return f"MaxPool\n{layer.kernel_size}×{layer.kernel_size}\nstride={layer.stride}"
                elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    return f"AvgPool\nAdaptive\n{layer.output_size}"
                else:
                    return name

            # 设置边的属性
            dot.attr('edge',
                color=colors['edge'],
                penwidth='1.5',
                arrowsize='1.0',
                arrowhead='vee',
                weight='1.2'
            )

            def add_layers(module, parent_name='', rank=''):
                """添加层并处理连接"""
                important_layers = (nn.Conv2d, nn.Linear, nn.MaxPool2d, 
                                  nn.AdaptiveAvgPool2d, SEBlock)
                
                # 添加输入节点
                if not prev_layer[0]:
                    dot.node('input', 'Input\n(3, 224, 224)', 
                            **get_node_style('input', is_first=True))
                    prev_layer[0] = 'input'

                nodes_in_rank = []
                for idx, (name, layer) in enumerate(module.named_children()):
                    layer_name = f"{parent_name}/{name}" if parent_name else name
                    
                    if isinstance(layer, important_layers):
                        # 创建节点
                        node_id = layer_name.replace('/', '_')
                        label = get_layer_label(layer, name)
                        style = get_node_style(layer.__class__.__name__)
                        
                        dot.node(node_id, label, **style)
                        nodes_in_rank.append(node_id)
                        
                        # 添加连接
                        if prev_layer[0]:
                            dot.edge(prev_layer[0], node_id)
                        prev_layer[0] = node_id

                    # 递归处理子层
                    if len(list(layer.children())) > 0:
                        add_layers(layer, layer_name, f"rank_{len(nodes_in_rank)}")

                # 使用相同的rank来对齐节点
                if nodes_in_rank:
                    with dot.subgraph() as s:
                        s.attr(rank='same')
                        for node in nodes_in_rank:
                            s.node(node)

                # 添加输出节点
                if not list(module.children()):
                    dot.node('output', 'Output\n(2)', 
                            **get_node_style('output', is_last=True))
                    if prev_layer[0]:
                        dot.edge(prev_layer[0], 'output')

            # 添加标题和图例
            dot.attr(label=f'\n{model_name.upper()} Neural Network Architecture\n', 
                    labelloc='t', 
                    fontsize='24',
                    fontcolor=colors['title'])

            # 生成主图
            prev_layer = [None]
            add_layers(model)
            
            # 保存为高质量PNG
            output_path = self.output_dir / f"{model_name}_architecture"
            dot.render(str(output_path), format='png', cleanup=True)
            
            return str(output_path) + '.png'
            
        except Exception as e:
            print(f"生成架构图失败: {str(e)}")
            if self.verbose:
                import traceback
                print(traceback.format_exc())
            return None

    def _get_model_info(self, model_path):
        """获取模型信息"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 获取准确率信息
        accuracy = checkpoint.get('current_acc', checkpoint.get('best_acc', 0))
        if 'best_model_' in model_path.name:
            accuracy = checkpoint.get('best_recorded_acc', checkpoint.get('best_acc', 0))
        
        return {
            'path': model_path,
            'filename': model_path.name,
            'model_name': checkpoint.get('model_name', 'unknown'),
            'accuracy': accuracy,
            'epoch': checkpoint.get('epoch', -1),
            'config': checkpoint.get('config', {})
        }

    def visualize_all(self):
        """为所有模型生成可视化"""
        if not self.available_models:
            print("未找到任何模型文件!")
            return

        print(f"\n找到 {len(self.available_models)} 个模型文件，开始生成架构图...")
        
        results = []
        for model in self.available_models:
            model_name = model['model_name']
            print(f"\n处理模型: {model_name}", end="... ", flush=True)
            
            png_path = self.generate_architecture(model['path'], model_name)
            if png_path:
                print("完成")
                results.append({
                    'model_name': model_name,
                    'accuracy': model['accuracy'],
                    'png_path': png_path
                })
            else:
                print("失败")

        if results:
            print("\n=== 生成结果 ===")
            headers = ["模型", "准确率", "架构图路径"]
            rows = [[r['model_name'], f"{r['accuracy']:.2f}%", r['png_path']] for r in results]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        print(f"\n架构图已保存至: {self.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='模型架构可视化工具')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='显示详细信息')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='指定输出目录')
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        visualizer = ModelVisualizer(verbose=args.verbose)
        
        if args.output:
            visualizer.output_dir = Path(args.output)
            visualizer.output_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer.visualize_all()
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        if args.verbose:
            import traceback
            print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
