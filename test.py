import os
import torch
import customtkinter as ctk
from PIL import Image, ImageTk
import torch.nn.functional as F
from torchvision import transforms
from models.models import get_model
from utils.config import Config

class ModelTester(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 确保使用正确的路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        config = Config()
        self.model_dir = os.path.join(self.base_dir, config.model_dir.lstrip('./'))
        
        if not os.path.exists(self.model_dir):
            print(f"错误：模型目录不存在: {self.model_dir}")
            print("尝试创建目录...")
            os.makedirs(self.model_dir, exist_ok=True)
        
        # 获取可用的模型列表
        self.available_models = self._get_available_models()
        if not self.available_models:
            print("\n错误：未找到任何训练好的模型!")
            print("搜索目录:", self.model_dir)
            print("\n当前目录下的文件:")
            for file in os.listdir(self.model_dir):
                print(f"- {file}")
            return
        
        # 设置窗口
        self.title("猫狗图像分类测试器 (Cat & Dog Classifier)")
        window_width = 1200
        window_height = 900
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # 初始化其他变量和UI
        self.current_image_path = None
        self.image_transform = self._get_transform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selected_model_info = None
        self.model = None
        
        # 创建UI
        self._setup_ui()
    
    def _get_transform(self):
        """获取图像预处理转换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_available_models(self):
        """获取所有可用的模型信息"""
        models = []
        
        print(f"\n正在搜索模型文件，目录: {self.model_dir}")
        
        # 扫描模型文件并获取信息
        for file in os.listdir(self.model_dir):
            if not file.endswith('.pth'):
                continue
                
            try:
                checkpoint_path = os.path.join(self.model_dir, file)
                print(f"发现模型文件: {file}")
                
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location='cpu',
                    weights_only=True
                )
                
                acc = checkpoint.get('best_acc', 0)
                model_name = checkpoint.get('model_name', '')
                
                if not model_name and file.startswith('best_model_'):
                    # 如果模型名称不在checkpoint中，从文件名提取
                    model_name = file.replace('best_model_', '').replace('.pth', '')
                
                if not model_name:
                    print(f"跳过文件 {file}: 无法确定模型名称")
                    continue
                
                # 将每个文件作为单独的模型条目
                model_info = {
                    'name': model_name,
                    'version': {
                        'type': 'best' if 'best_model_' in file else 'last',
                        'accuracy': acc,
                        'file': file
                    }
                }
                models.append(model_info)
                print(f"添加模型: {model_name} ({model_info['version']['type']}, acc={acc:.2f}%)")
                
            except Exception as e:
                print(f"读取模型文件 {file} 出错: {str(e)}")
        
        # 按准确率排序
        return sorted(models, key=lambda x: (-x['version']['accuracy'], x['version']['file']))
    
    def _setup_ui(self):
        """设置用户界面"""
        # 主框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        # 左侧面板 - 模型选择区域
        self.left_panel = ctk.CTkFrame(self.main_frame, width=300)
        self.left_panel.pack(side="left", fill="both", padx=20, pady=20)
        
        # 标题
        title_label = ctk.CTkLabel(
            self.left_panel,
            text="模型选择\nModel Selection",
            font=("Arial", 20, "bold"),
            justify="center"
        )
        title_label.pack(pady=(0, 20))
        
        # 模型列表框架
        self.models_frame = ctk.CTkScrollableFrame(
            self.left_panel,
            width=280,
            height=400,
            label_text="可用模型文件 / Available Model Files"
        )
        self.models_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 添加所有模型文件
        for model_info in self.available_models:
            model_btn = ctk.CTkButton(
                self.models_frame,
                text=(f"{model_info['name'].upper()}\n"
                      f"类型: {'最佳' if model_info['version']['type']=='best' else '最新'}\n"
                      f"文件: {model_info['version']['file']}\n"
                      f"准确率: {model_info['version']['accuracy']:.2f}%"),
                command=lambda m=model_info: self._on_model_selected(m),
                width=250,
                height=80,
                font=("Arial", 14),
                fg_color="#2B6B99" if model_info['version']['type']=='best' else "#407294",
                hover_color="#1E4B6C"
            )
            model_btn.pack(pady=5)
        
        # 加载模型按钮
        self.load_button = ctk.CTkButton(
            self.left_panel,
            text="加载选中的模型\nLoad Selected Model",
            command=self._load_selected_model,
            width=200,
            height=60,
            font=("Arial", 16, "bold"),
            fg_color="#28794C",
            hover_color="#1B5434",
            state="disabled"
        )
        self.load_button.pack(pady=20)
        
        # 当前模型信息显示
        self.model_info = ctk.CTkTextbox(
            self.left_panel,
            width=250,
            height=120,
            font=("Arial", 14),
            wrap="word"
        )
        self.model_info.pack(pady=10)
        self.model_info.insert("1.0", "请选择要加载的模型\nPlease select a model to load")
        self.model_info.configure(state="disabled")
        
        # 右侧面板
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        # 图片显示区域
        self.image_frame = ctk.CTkFrame(self.right_panel)
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text="请选择图片\nPlease Select an Image",
            font=("Arial", 18),
            width=450,
            height=450
        )
        self.image_label.pack(expand=True)
        
        # 按钮区域
        self.button_frame = ctk.CTkFrame(self.right_panel)
        self.button_frame.pack(fill="x", pady=20)
        
        self.select_button = ctk.CTkButton(
            self.button_frame,
            text="选择图片\nSelect Image",
            command=self._select_image,
            width=200,
            height=50,
            font=("Arial", 16),
            fg_color="#28794C",  # 绿色
            hover_color="#1B5434"  # 深绿色
        )
        self.select_button.pack(side="left", expand=True, padx=10)
        
        self.predict_button = ctk.CTkButton(
            self.button_frame,
            text="开始预测\nPredict",
            command=self._predict_image,
            width=200,
            height=50,
            font=("Arial", 16),
            fg_color="#B93A3A",  # 红色
            hover_color="#822929",  # 深红色
            state="disabled"
        )
        self.predict_button.pack(side="right", expand=True, padx=10)
        
        # 预测结果区域
        self.result_text = ctk.CTkTextbox(
            self.right_panel,
            height=150,
            font=("Arial", 16),
            wrap="word"
        )
        self.result_text.pack(fill="x", padx=20, pady=10)
        self.result_text.insert("1.0", "预测结果将在这里显示\nPrediction results will be shown here")
        self.result_text.configure(state="disabled")
        
        # 置信度区域
        self.confidence_frame = ctk.CTkFrame(self.right_panel)
        self.confidence_frame.pack(fill="x", padx=20, pady=10)
        
        self.confidence_label = ctk.CTkLabel(
            self.confidence_frame,
            text="置信度 / Confidence:",
            font=("Arial", 16)
        )
        self.confidence_label.pack(side="left", padx=10)
        
        self.confidence_bar = ctk.CTkProgressBar(
            self.confidence_frame,
            width=400,
            height=20,
            progress_color="#2B6B99"
        )
        self.confidence_bar.pack(side="left", padx=10, pady=10)
        self.confidence_bar.set(0)
        
        # 状态栏
        self.status_label = ctk.CTkLabel(
            self.right_panel,
            text="就绪 / Ready",
            font=("Arial", 14)
        )
        self.status_label.pack(pady=10)

    def _select_image(self):
        """选择图片"""
        file_path = ctk.filedialog.askopenfilename(
            initialdir="./train",
            title="选择图片",
            filetypes=(
                ("图片文件", "*.jpg *.jpeg *.png"),
                ("所有文件", "*.*")
            )
        )
        
        if file_path:
            self.current_image_path = file_path
            self._display_image(file_path)
            self.predict_button.configure(state="normal")
            self.status_label.configure(text="已加载图片，可以开始预测")
    
    def _display_image(self, image_path):
        """显示图片"""
        try:
            # 加载并调整图片大小
            image = Image.open(image_path)
            image = image.resize((350, 350), Image.Resampling.LANCZOS)
            
            # 使用CTkImage替代PhotoImage
            photo = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=(350, 350)
            )
            
            # 更新显示（移除文字）
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            self.status_label.configure(text=f"图片加载错误 / Image Error: {str(e)}")
    
    def _on_model_selected(self, model_info):
        """当选择模型时"""
        self.selected_model_info = model_info
        
        # 更新模型信息显示
        info_text = (
            f"已选择模型 / Selected Model:\n"
            f"{model_info['name'].upper()}\n"
            f"文件 / File: {model_info['version']['file']}\n"
            f"类型 / Type: "
            f"{'最佳 / Best' if model_info['version']['type']=='best' else '最新 / Latest'}\n"
            f"准确率 / Accuracy: {model_info['version']['accuracy']:.2f}%"
        )
        self.model_info.configure(state="normal")
        self.model_info.delete("1.0", "end")
        self.model_info.insert("1.0", info_text)
        self.model_info.configure(state="disabled")
        
        # 启用加载按钮
        self.load_button.configure(state="normal")
    
    def _load_selected_model(self):
        """加载选中的模型"""
        if not self.selected_model_info:
            return
            
        try:
            checkpoint_path = os.path.join(
                self.model_dir,
                self.selected_model_info['version']['file']
            )
            
            # 创建模型实例
            self.model = get_model(
                self.selected_model_info['name'],
                pretrained=False
            ).to(self.device)
            
            # 加载模型权重
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # 设置为评估模式
            
            # 更新状态
            self.predict_button.configure(
                state="normal" if self.current_image_path else "disabled"
            )
            
            # 更新提示信息
            success_msg = (
                f"模型已加载 / Model loaded: {self.selected_model_info['name'].upper()}\n"
                f"准确率 / Accuracy: {self.selected_model_info['version']['accuracy']:.2f}%"
            )
            self.status_label.configure(text=success_msg)
            
            # 重置结果显示
            self._update_result_text(
                "模型已加载，请选择图片进行预测\n"
                "Model loaded, please select an image to predict"
            )
            
        except Exception as e:
            error_msg = f"加载模型出错 / Error loading model: {str(e)}"
            print(error_msg)  # 打印详细错误信息
            self.status_label.configure(text=error_msg)
            self._update_result_text("模型加载失败，请检查控制台输出")
    
    def _update_result_text(self, text):
        """更新结果文本显示"""
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")
    
    @torch.no_grad()
    def _predict_image(self):
        """预测图片"""
        if not self.current_image_path or not self.model:
            return
        
        try:
            # 加载和预处理图片
            image = Image.open(self.current_image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # 设置为评估模式
            self.model.eval()
            
            # 预测
            outputs = self.model(image_tensor)
            
            # 注意：我们的模型输出可能已经经过log_softmax，需要先转换
            if outputs.size(1) == 2:  # 如果是二分类问题
                if self.selected_model_info['name'] in ['resnet', 'alexnet', 'squeezenet']:
                    # 这些模型使用了log_softmax，需要先exp
                    probs = torch.exp(outputs)
                else:
                    # 直接使用softmax
                    probs = F.softmax(outputs, dim=1)
            
            pred_prob, pred_class = probs.max(1)
            
            # 获取结果
            pred_prob = pred_prob.item()
            is_dog = pred_class.item() == 1
            
            # 更新结果显示
            result_text = (
                f"预测结果 / Prediction Results:\n\n"
                f"使用模型 / Model: {self.selected_model_info['name'].upper()}\n"
                f"预测类别 / Class: {'狗 / Dog' if is_dog else '猫 / Cat'}\n"
                f"置信度 / Confidence: {pred_prob:.2%}\n"
                f"\n详细概率:\n"
                f"猫 / Cat: {probs[0][0].item():.2%}\n"
                f"狗 / Dog: {probs[0][1].item():.2%}\n"
                f"\n图片路径 / Image Path:\n{os.path.basename(self.current_image_path)}"
            )
            self._update_result_text(result_text)
            
            # 更新UI
            self.confidence_bar.set(pred_prob)
            self.status_label.configure(
                text=f"预测完成! 类别: {'狗 / Dog' if is_dog else '猫 / Cat'} "
                     f"置信度: {pred_prob:.2%}"
            )
            
        except Exception as e:
            error_msg = f"预测出错: {str(e)}"
            print(error_msg)  # 打印详细错误信息
            self.status_label.configure(text=error_msg)
            self._update_result_text("预测过程中出现错误，请检查控制台输出")

if __name__ == "__main__":
    try:
        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 初始化配置
        config = Config()
        model_dir = os.path.join(current_dir, config.model_dir.lstrip('./'))
        
        print(f"\n当前工作目录: {os.getcwd()}")
        print(f"脚本所在目录: {current_dir}")
        print(f"模型目录: {model_dir}")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"\n创建模型目录: {model_dir}")
        
        # 检查是否有模型文件
        model_files = [f for f in os.listdir(model_dir) 
                      if f.endswith('.pth')]
        
        if model_files:
            print("\n发现以下模型文件:")
            for file in model_files:
                print(f"- {file}")
            
            # 启动应用
            app = ModelTester()
            if hasattr(app, 'available_models'):
                app.mainloop()
            else:
                print("\n无法正确初始化应用，请检查上述错误信息。")
                input("\n按回车键退出...")
        else:
            print(f"\n错误：在 {model_dir} 中未找到任何.pth模型文件!")
            print("\n期望的文件格式:")
            print("1. last_model.pth - 最新训练的模型")
            print("2. best_model_xxx.pth - 最佳性能的模型")
            input("\n按回车键退出...")
            
    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        print("错误类型:", e.__class__.__name__)
        input("\n按回车键退出...")
