import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_processor import AudioProcessor, TimbreDataset
from model import TimbreClassifier
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import json
import os

class TimbreTrainer:
    def __init__(
        self,
        model: TimbreClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for features, labels in tqdm(self.train_loader, desc="Training"):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            thickness_logits, brightness_logits, solidity_logits = self.model(features)
            
            # 计算损失
            loss = (
                self.criterion(thickness_logits, labels[:, 0]) +
                self.criterion(brightness_logits, labels[:, 1]) +
                self.criterion(solidity_logits, labels[:, 2])
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self) -> Tuple[float, List[float]]:
        self.model.eval()
        total_loss = 0
        accuracies = [0, 0, 0]  # 三个指标的准确率
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Evaluating"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                thickness_logits, brightness_logits, solidity_logits = self.model(features)
                
                # 计算损失
                loss = (
                    self.criterion(thickness_logits, labels[:, 0]) +
                    self.criterion(brightness_logits, labels[:, 1]) +
                    self.criterion(solidity_logits, labels[:, 2])
                )
                total_loss += loss.item()
                
                # 计算准确率
                thickness_pred = torch.argmax(thickness_logits, dim=-1)
                brightness_pred = torch.argmax(brightness_logits, dim=-1)
                solidity_pred = torch.argmax(solidity_logits, dim=-1)
                
                accuracies[0] += (thickness_pred == labels[:, 0]).float().mean().item()
                accuracies[1] += (brightness_pred == labels[:, 1]).float().mean().item()
                accuracies[2] += (solidity_pred == labels[:, 2]).float().mean().item()
        
        return total_loss / len(self.val_loader), [acc / len(self.val_loader) for acc in accuracies]
    
    def train(self, num_epochs: int, save_dir: str):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")
            
            # 评估
            val_loss, accuracies = self.evaluate()
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Accuracies - Thickness: {accuracies[0]:.4f}, Brightness: {accuracies[1]:.4f}, Solidity: {accuracies[2]:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                
                # 保存训练信息
                info = {
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "accuracies": accuracies
                }
                with open(os.path.join(save_dir, "best_model_info.json"), "w") as f:
                    json.dump(info, f)
            
            # 更新学习率
            self.scheduler.step()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建保存目录
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化数据处理器
    processor = AudioProcessor()
    
    # TODO: 加载数据集
    # 这里需要根据实际数据集路径进行修改
    train_audio_paths = []  # 训练集音频路径列表
    train_labels = []       # 训练集标签列表
    val_audio_paths = []    # 验证集音频路径列表
    val_labels = []         # 验证集标签列表
    
    # 创建数据集
    train_dataset = TimbreDataset(train_audio_paths, train_labels, processor)
    val_dataset = TimbreDataset(val_audio_paths, val_labels, processor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = TimbreClassifier()
    
    # 初始化训练器
    trainer = TimbreTrainer(model, train_loader, val_loader)
    
    # 开始训练
    trainer.train(num_epochs=100, save_dir=save_dir)

if __name__ == "__main__":
    main() 