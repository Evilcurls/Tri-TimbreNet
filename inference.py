import torch
from model import TimbreClassifier
from data_processor import AudioProcessor
import os
import json
from typing import Tuple, List
import numpy as np

class TimbrePredictor:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = AudioProcessor()
        self.model = TimbreClassifier()
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
    
    def predict_audio(self, audio_path: str) -> Tuple[int, int, int]:
        """预测单个音频文件的三个指标"""
        # 加载和预处理音频
        waveform = self.processor.load_audio(audio_path)
        segments = self.processor.segment_audio(waveform)
        features = self.processor.extract_features(segments)
        
        # 将特征移到设备上
        features = features.to(self.device)
        
        # 对每个片段进行预测
        predictions = []
        with torch.no_grad():
            for segment_features in features:
                thickness, brightness, solidity = self.model.predict(segment_features.unsqueeze(0))
                predictions.append((thickness, brightness, solidity))
        
        # 使用投票机制整合预测结果
        final_thickness = self._majority_vote([p[0] for p in predictions])
        final_brightness = self._majority_vote([p[1] for p in predictions])
        final_solidity = self._majority_vote([p[2] for p in predictions])
        
        return final_thickness, final_brightness, final_solidity
    
    def _majority_vote(self, predictions: List[int]) -> int:
        """使用多数投票机制确定最终预测结果"""
        return max(set(predictions), key=predictions.count)
    
    def predict_batch(self, audio_paths: List[str]) -> List[Tuple[int, int, int]]:
        """批量预测多个音频文件"""
        return [self.predict_audio(path) for path in audio_paths]

def main():
    # 模型路径
    model_path = "checkpoints/best_model.pth"
    
    # 初始化预测器
    predictor = TimbrePredictor(model_path)
    
    # 测试音频路径
    test_audio_path = "path/to/test/audio.wav"
    
    # 进行预测
    thickness, brightness, solidity = predictor.predict_audio(test_audio_path)
    
    print(f"预测结果:")
    print(f"厚薄: {thickness}")
    print(f"明暗: {brightness}")
    print(f"虚实: {solidity}")

if __name__ == "__main__":
    main() 