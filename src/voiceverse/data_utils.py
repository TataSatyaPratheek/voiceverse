import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pandas as pd
import librosa
from typing import List, Dict, Tuple, Optional
import re
import json

class TeluguEnglishDataset(Dataset):
    """Dataset for Telugu-English mixed TTS training"""
    
    def __init__(self, data_dir: Path, metadata_file: str, 
                 sample_rate: int = 22050, max_length: int = 8.0):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        # Load metadata
        self.metadata = pd.read_csv(self.data_dir / metadata_file)
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self) -> List[Dict]:
        """Prepare training samples with text and audio paths"""
        samples = []
        for _, row in self.metadata.iterrows():
            sample = {
                'audio_path': self.data_dir / 'audio' / f"{row['file_id']}.wav",
                'text': row['text'],
                'language': self._detect_language(row['text']),
                'speaker_id': row['speaker_id'],
                'emotion': row.get('emotion', 'neutral')
            }
            samples.append(sample)
        return samples
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Telugu, English, or mixed"""
        # Telugu Unicode range: \u0C00-\u0C7F
        telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if telugu_chars > english_chars:
            return 'telugu'
        elif english_chars > telugu_chars:
            return 'english'
        else:
            return 'mixed'
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load and process audio
        audio, sr = librosa.load(sample['audio_path'], sr=self.sample_rate)
        
        # Trim/pad to max_length
        max_samples = int(self.max_length * self.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Extract mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=80, hop_length=256, win_length=1024)
        mel_db = librosa.power_to_db(mel)
        
        # Text preprocessing
        processed_text = self._preprocess_text(sample['text'])
        
        return {
            'audio': torch.FloatTensor(audio),
            'mel': torch.FloatTensor(mel_db),
            'text': processed_text,
            'language': sample['language'],
            'speaker_id': torch.LongTensor([hash(sample['speaker_id']) % 1000]),
            'emotion': sample['emotion']
        }
    
    def _preprocess_text(self, text: str) -> torch.LongTensor:
        """Convert text to token sequence"""
        # Implement phoneme conversion for Telugu and English
        # This should handle mixed scripts and code-switching
        tokens = [ord(c) % 1000 for c in text[:100]]  # placeholder
        tokens = tokens + [0] * (100 - len(tokens))  # pad to fixed length
        return torch.LongTensor(tokens)

def create_persona_calibration_dataset(persona_name: str, 
                                     reference_clips: List[Path],
                                     target_style: Dict[str, float]) -> Dataset:
    """Create dataset for persona vector calibration"""
    
    class PersonaCalibrationDataset(Dataset):
        def __init__(self):
            self.clips = reference_clips
            self.target_style = target_style
            
        def __len__(self):
            return len(self.clips)
            
        def __getitem__(self, idx):
            clip_path = self.clips[idx]
            audio, sr = librosa.load(clip_path, sr=22050)
            
            # Extract style features
            features = self._extract_style_features(audio, sr)
            
            return {
                'audio': torch.FloatTensor(audio),
                'style_features': torch.FloatTensor(features),
                'target_style': torch.FloatTensor(list(self.target_style.values()))
            }
        
        def _extract_style_features(self, audio, sr):
            """Extract tempo, energy, pitch characteristics"""
            # F0 estimation
            f0 = librosa.yin(audio, fmin=80, fmax=400, sr=sr)
            f0_mean = np.nanmean(f0)
            f0_std = np.nanstd(f0)
            
            # Energy
            rms = librosa.feature.rms(y=audio)[0]
            energy_mean = np.mean(rms)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            warmth = np.mean(spectral_centroid)
            
            return [f0_mean, f0_std, energy_mean, warmth]
    
    return PersonaCalibrationDataset()
