# 基于深度学习的音色分类系统

本项目实现了一个基于wav2vec2.0的音色分类系统，用于评估音频的三个关键音色特征：厚薄、明暗和虚实。每个特征都有5个等级的分类。

## 项目结构

```
.
├── data_processor.py    
├── model.py            
├── train.py           
├── inference.py       
├── requirements.txt   
└── README.md         
```

## 环境要求

- Python 3.8+
- PyTorch 1.9.0+
- CUDA 

## 安装

1. 克隆项目：
```bash
git clone https://github.com/Evilcurls/Tri-TimbreNet.git
cd Tri-TimbreNet
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

下载cuc-ted（Comunication University of China Timbre evaluation dataset)

解压至data文件夹

## 训练模型

1. 设置`train.py` 中的训练集和验证集的路径位置

2. 开始训练：
```bash
python train.py
```

训练过程中会自动保存最佳模型到 `checkpoints` 目录。

## 模型推理

1. 准备测试音频文件

2. 运行推理：
```bash
python inference.py
```

- 