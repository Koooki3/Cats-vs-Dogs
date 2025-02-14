# Cats vs Dogs Classification with PyTorch

This repository contains code and resources for building, training, and evaluating deep learning models for cat vs dog classification using PyTorch.

本仓库包含使用 PyTorch 构建、训练和评估猫狗分类深度学习模型的代码和资源。

## Table of Contents / 目录
- [Introduction / 简介](#introduction--简介)
- [Installation / 安装](#installation--安装)
- [Usage / 使用方法](#usage--使用方法)
- [Contributing / 贡献](#contributing--贡献)

## Introduction / 简介
Cat vs Dog classification is a binary classification task that distinguishes between cat and dog images. This project demonstrates various deep learning models and visualization tools using PyTorch.

猫狗分类是一个二分类任务，通过图像区分猫和狗。本项目演示了使用 PyTorch 搭建和训练多种模型以及相应的架构可视化工具。

## Installation / 安装
To get started, clone the repository and install the required dependencies:

首先，克隆仓库并安装所需的依赖项：

```bash
git clone https://github.com/Koooki3/Cats-vs-Dogs.git
cd Cats-vs-Dogs
pip install -r requirements.txt
```

## Usage / 使用方法
### Data Preparation / 数据准备
To process the graphs dataset, make sure you prepare necessary data and settle them in "train" with following name rules: "X(nums)_Categorys.png/jpg/..."

要处理图像数据，请确保必要的数据被保存在用户自行创建的“train”文件夹内且内部文件命名为："X(序号)_种类.png/jpg/..."

In order to create dataset, you can use codes in:

你可以使用压缩包中的代码帮助你创建数据集：

```
ZIP file: data-preparation.zip
```

### Training / 训练
To train a model, run the following command:

要训练一个模型，请运行以下命令：

```bash
python train.py
```

### Evaluation / 评估
To evaluate the trained model, run the following command:

要评估训练好的模型，请运行以下命令：

```bash
python chech_models.py
python visualize_models.py
```

### Test / 测试
To test all the models which have been trained, run the following command. By using the created GUI table, it will be pretty easy for you to check all models and test whatever you want. 

要测试所有训练好的模型，请运行以下命令。通过使用脚本生成的GUI界面，你将能轻松查看所有模型的信息并测试任何一个模型。

```bash
python test.py
```

### Notice / 注意项
The project could be easily transform to solve other Binary-Classification problems, just make sure data in "train" is properly set-up.

这个项目可以轻松迁移用以训练不同的二分类任务，只需保证“train”中的数据满足要求即可。

## Contributing / 贡献
Contributions are welcome! Please open an issue or submit a pull request for any changes.

欢迎贡献！请提交 issue 或 pull request 以进行任何更改。
