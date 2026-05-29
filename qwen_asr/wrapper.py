import torch
import torch.nn as nn

import torchaudio

from .embedding import Qwen3ASREmbeddingExtractor

class Qwen3ASRWrapper(nn.Module):
    def __init__(self, model_path='Qwen/Qwen3-ASR-0.6B') -> None:
        super().__init__()
        self.model = Qwen3ASREmbeddingExtractor.from_pretrained(model_path, dtype=torch.bfloat16)
        self.sample_rate = 16000
        
    @ torch.inference_mode()
    def forward(self, audio: torch.Tensor, sample_rate: int, add_extra_padding: bool = True) -> torch.Tensor:
        device = next(self.model.audio_tower.parameters()).device
        
        audio = audio[:1, ...]
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
        audio = audio.to(device)

        if add_extra_padding:
            audio = torch.nn.functional.pad(audio, (0, 1231), value=0)

        result = self.model.extract(audio)
        return result.embedding.transpose(0, 1) # [c, t]