from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class VoiceVerseConfig:
    # Model paths
    styletts_checkpoint: str = "tech/StyleTTS2/checkpoints/Models/LJSpeech/epoch_2nd_00180.pth"
    styletts_config: str = "tech/StyleTTS2/checkpoints/Models/LJSpeech/config.yml"
    hifigan_checkpoint: str = "tech/vocoder_ckpt/bigvgan_generator.pt"
    hifigan_config: str = "tech/vocoder_ckpt/config.json"

    # Aux components required to build StyleTTS2
    asr_weights: str = "tech/StyleTTS2/Utils/ASR/epoch_00080.pth"
    asr_config: str = "tech/StyleTTS2/Utils/ASR/config.yml"
    f0_path: str = "tech/StyleTTS2/Utils/JDC/bst.t7"
    plbert_dir: str = "tech/StyleTTS2/Utils/PLBERT"
    # You may need to add others depending on model setup

    # Training parameters
    batch_size: int = 4
    learning_rate: float = 0.001
    epochs: int = 50

    # Quality thresholds
    quality_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                'min_mos': 3.0,
                'min_snr': 0.6, 
                'min_pronunciation': 0.8,
                'min_style_consistency': 0.75
            }

# Global config instance
config = VoiceVerseConfig()
