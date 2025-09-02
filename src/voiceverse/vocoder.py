# src/voiceverse/vocoder.py
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional

class HiFiGANVocoder:
    """
    HiFi-GAN interface; can be replaced with a trained generator compatible with your mel config.
    """
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self.model = self._load_hifigan()

    def _load_hifigan(self):
        # Example: if you have a generator class; adapt to your real implementation
        # Fallback: use BigVGAN if present as a drop-in
        try:
            from bigvgan import BigVGAN  # if installed from NVIDIA repo or your local folder
            with open(self.config_path) as f:
                h = json.load(f)
            model = BigVGAN.from_hparams(hparams=h) if hasattr(BigVGAN, "from_hparams") else BigVGAN(h)
            state = torch.load(self.checkpoint_path, map_location="cpu")
            # Common key is 'generator'
            if "generator" in state:
                model.load_state_dict(state["generator"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            model.remove_weight_norm() if hasattr(model, "remove_weight_norm") else None
            return model.to(self.device).eval()
        except Exception as e:
            print(f"[HiFiGAN] Falling back due to import/load error: {e}")
            return None

    def vocode(self, mel: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        if self.model is None:
            # last-resort Griffin-Lim fallback to keep pipeline alive
            import librosa
            mel_db = mel
            S = librosa.db_to_power(mel_db)
            audio = librosa.feature.inverse.mel_to_audio(S, sr=sample_rate, n_iter=32)
            return audio.astype(np.float32)
        mel_t = torch.tensor(mel, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            wav = self.model(mel_t)
            wav = wav.squeeze().detach().cpu().float().numpy()
        return wav


class BigVGANVocoder:
    """
    BigVGAN v2 interface using your local vocoder_ckpt generator checkpoint and code.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda", use_cuda_kernel: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.use_cuda_kernel = use_cuda_kernel
        self.model = self._load_bigvgan()

    def _ensure_bigvgan_on_path(self):
        # Try local vendor locations
        local_vendor = Path("tech/GPT-SoVITS/BigVGAN").resolve()
        if local_vendor.exists() and str(local_vendor) not in sys.path:
            sys.path.insert(0, str(local_vendor))
        local_ckpt_dir = Path("tech/vocoder_ckpt").resolve()
        if local_ckpt_dir.exists() and str(local_ckpt_dir) not in sys.path:
            sys.path.insert(0, str(local_ckpt_dir))

    def _load_bigvgan(self):
        self._ensure_bigvgan_on_path()
        try:
            from bigvgan import BigVGAN
            # If you have a JSON config, you can pass it; otherwise construct with defaults
            model = BigVGAN(h=None, use_cuda_kernel=self.use_cuda_kernel)
            state = torch.load(self.checkpoint_path, map_location="cpu")
            # Usually generator weights at top-level or under 'generator'
            if "generator" in state:
                model.load_state_dict(state["generator"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            if hasattr(model, "remove_weight_norm"):
                model.remove_weight_norm()
            return model.to(self.device).eval()
        except Exception as e:
            print(f"[BigVGAN] Load error: {e}")
            return None

    def vocode(self, mel: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        if self.model is None:
            # Griffn-Lim fallback (keeps flow alive; replace as you stabilize vocoder)
            import librosa
            S = librosa.db_to_power(mel)
            audio = librosa.feature.inverse.mel_to_audio(S, sr=sample_rate, n_iter=32)
            return audio.astype(np.float32)
        mel_t = torch.tensor(mel, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            wav = self.model(mel_t)  # expected shape [B, 1, T]
            wav = wav.squeeze().detach().cpu().float().numpy()
        return wav
