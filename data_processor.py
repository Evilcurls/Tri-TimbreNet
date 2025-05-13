import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from typing import List, Tuple

class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, segment_length: int = 3, overlap: float = 0.5):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.overlap = overlap
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """加载音频文件并重采样"""
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()
    
    def segment_audio(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """将音频分段"""
        segment_samples = int(self.segment_length * self.sample_rate)
        overlap_samples = int(segment_samples * self.overlap)
        stride = segment_samples - overlap_samples
        
        segments = []
        for i in range(0, len(waveform) - segment_samples + 1, stride):
            segment = waveform[i:i + segment_samples]
            segments.append(segment)
        return segments
    
    def extract_features(self, segments: List[torch.Tensor]) -> torch.Tensor:
        """提取音频特征"""
        features = []
        for segment in segments:
            inputs = self.feature_extractor(
                segment, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            features.append(inputs.input_values.squeeze())
        return torch.stack(features)

class TimbreDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths: List[str], labels: List[Tuple[int, int, int]], processor: AudioProcessor):
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.audio_paths[idx]
        waveform = self.processor.load_audio(audio_path)
        segments = self.processor.segment_audio(waveform)
        features = self.processor.extract_features(segments)
        
        # 将标签转换为one-hot编码
        thickness, brightness, solidity = self.labels[idx]
        labels = torch.tensor([
            thickness - 1,  # 转换为0-4的索引
            brightness - 1,
            solidity - 1
        ])
        
        return features, labels 