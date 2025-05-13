import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from typing import Tuple

class TimeAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x, x, x)[0]

class FeatureFusion(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(x)

class TimbreClassifier(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.hidden_size = self.wav2vec2.config.hidden_size
        
        # 特征增强模块
        self.time_attention = TimeAttention(self.hidden_size)
        self.feature_fusion = FeatureFusion(self.hidden_size)
        
        # 分类头
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            ) for _ in range(3)  # 三个指标的分类器
        ])
        
    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 获取wav2vec2特征
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        
        # 特征增强
        attended = self.time_attention(hidden_states)
        enhanced = self.feature_fusion(attended)
        
        # 全局平均池化
        pooled = torch.mean(enhanced, dim=1)
        
        # 分类
        thickness_logits = self.classifier[0](pooled)
        brightness_logits = self.classifier[1](pooled)
        solidity_logits = self.classifier[2](pooled)
        
        return thickness_logits, brightness_logits, solidity_logits
    
    def predict(self, input_values: torch.Tensor) -> Tuple[int, int, int]:
        """预测单个样本的三个指标"""
        thickness_logits, brightness_logits, solidity_logits = self(input_values)
        
        thickness_pred = torch.argmax(thickness_logits, dim=-1).item() + 1
        brightness_pred = torch.argmax(brightness_logits, dim=-1).item() + 1
        solidity_pred = torch.argmax(solidity_logits, dim=-1).item() + 1
        
        return thickness_pred, brightness_pred, solidity_pred 