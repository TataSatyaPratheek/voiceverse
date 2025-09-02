"""VoiceVerse TTS Pipeline with Telugu/English support"""

from .tts import StyleTTSWrapper, ParlerTTSWrapper
from .vocoder import HiFiGANVocoder, BigVGANVocoder  
from .persona import PersonaVectorManager, PersonaConfig
from .quality import AudioQualityAssessor, QualityGate, QualityMetrics
from .data_utils import TeluguEnglishDataset
from .training import PersonaFineTuner, AdaptivePersonaTrainer
from .pipeline import episode_pipeline

__version__ = "0.1.0"
__all__ = [
    "StyleTTSWrapper",
    "ParlerTTSWrapper",
    "HiFiGANVocoder",
    "BigVGANVocoder",
    "PersonaVectorManager",
    "PersonaConfig",
    "AudioQualityAssessor",
    "QualityGate",
    "QualityMetrics",
    "TeluguEnglishDataset",
    "PersonaFineTuner",
    "AdaptivePersonaTrainer",
    "episode_pipeline",
]