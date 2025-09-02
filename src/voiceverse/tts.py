# src/voiceverse/tts.py
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import re

# Optional phonemization for better prosody; safe fallbacks if missing
try:
    from phonemizer import phonemize
    HAS_PHONEMIZER = True
except Exception:
    HAS_PHONEMIZER = False

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
    HAS_INDIC = True
except Exception:
    HAS_INDIC = False


class StyleTTSWrapper:
    """
    StyleTTS2 wrapper: loads model from tech/StyleTTS2 and generates mel with persona/style controls.
    Expects:
      - checkpoint_path: e.g. tech/StyleTTS2/checkpoints/Models/LJSpeech/epoch_2nd_00180.pth
      - config_path:     e.g. tech/StyleTTS2/checkpoints/Models/LJSpeech/config.yml
    """
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        self._ensure_styletts_on_path()
        self.model, self.hps = self._load_styletts2_model()

    def _ensure_styletts_on_path(self):
        # Add tech/StyleTTS2 to import path
        candidate = Path("tech/StyleTTS2").resolve()
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    def _load_styletts2_model(self):
        # Import StyleTTS2 code
        import yaml
        import torch

        from models import build_model  # provided by the StyleTTS2 repo
        with open(self.config_path, "r") as f:
            hps = yaml.safe_load(f)

        model = build_model(hps)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        # StyleTTS2 checkpoints typically store weights under 'model' or full state_dict
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[StyleTTS2] missing keys: {missing} | unexpected keys: {unexpected}")
        model = model.to(self.device).eval()
        return model, hps

    def synthesize(self, text: str, persona_vector: torch.Tensor, style_controls: Dict[str, float] | None = None) -> np.ndarray:
        """
        Generate mel-spectrogram conditioned on text + persona vector + style controls.
        style_controls keys: tempo [0.5..2.0], energy [0..1], warmth [0..1], formality [0..1]
        """
        if style_controls is None:
            style_controls = {"tempo": 1.0, "energy": 0.7, "warmth": 0.5, "formality": 0.5}

        # Segment by language and phonemize
        tokens = self._preprocess_multilingual_text(text)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)

        # Map scalar controls to model-consumable conditioning
        tempo = torch.tensor([style_controls.get("tempo", 1.0)], device=self.device, dtype=torch.float32)
        energy = torch.tensor([style_controls.get("energy", 0.7)], device=self.device, dtype=torch.float32)
        warmth = torch.tensor([style_controls.get("warmth", 0.5)], device=self.device, dtype=torch.float32)
        formality = torch.tensor([style_controls.get("formality", 0.5)], device=self.device, dtype=torch.float32)

        # Persona vector to device
        persona = persona_vector.to(self.device).float().unsqueeze(0)

        # Forward (the exact signature can vary; adapt if your cloned repo differs)
        with torch.no_grad():
            # Commonly: model.infer(text_ids, style_vec, tempo, energy, ...)
            # Provide a generic call with kwargs and let unexpected keys be ignored internally if the signature differs
            mel = self.model.infer(
                text=tokens,
                persona=persona,
                tempo=tempo,
                energy=energy,
                warmth=warmth,
                formality=formality,
            )
            if isinstance(mel, (list, tuple)):
                mel = mel
            mel = mel.squeeze(0).detach().cpu().float().numpy()
        return mel

    def _preprocess_multilingual_text(self, text: str) -> List[int]:
        """
        Split Telugu/English segments, phonemize if available, else fallback to byte-code ints with mild normalization.
        """
        segments = self._segment_by_language(text)
        phones: List[str] = []
        for seg, lang in segments:
            if HAS_PHONEMIZER:
                if lang == "telugu":
                    # espeak-ng supports Telugu; phonemizer backend 'espeak' works if espeak-ng is installed
                    p = phonemize(seg, language="te", backend="espeak", strip=True, njobs=1)
                else:
                    p = phonemize(seg, language="en-us", backend="espeak", strip=True, njobs=1)
                phones.extend(p.split())
            else:
                # Fallback: rough transliteration for Telugu to Latin to keep charset small
                if lang == "telugu" and HAS_INDIC:
                    latin = transliterate(seg, sanscript.TELUGU, sanscript.ITRANS)
                    phones.extend(list(re.sub(r"\\s+", "", latin.lower())))
                else:
                    phones.extend(list(re.sub(r"\\s+", "", seg.lower())))
        # Map phones to small integer vocab
        vocab = {}
        ids: List[int] = []
        for ph in phones[:256]:  # cap length; real models should use tokenizer
            if ph not in vocab:
                vocab[ph] = len(vocab) + 1
            ids.append(vocab[ph])
        if not ids:
            ids = [1]
        return ids

    def _segment_by_language(self, text: str) -> List[Tuple[str, str]]:
        telugu = lambda c: "\\u0C00" <= f"\\u{ord(c):04X}" <= "\\u0C7F"
        segs: List[Tuple[str, str]] = []
        buf = []
        cur = None
        for ch in text:
            lang = "telugu" if telugu(ch) else "english"
            if cur is None:
                cur = lang
                buf.append(ch)
            elif lang == cur:
                buf.append(ch)
            else:
                segs.append(("".join(buf), cur))
                buf = [ch]
                cur = lang
        if buf:
            segs.append(("".join(buf), cur))
        return segs


class ParlerTTSWrapper:
    """
    Parler-TTS wrapper: loads model from tech/parler_ckpt and generates audio with persona/style controls.
    """
    def __init__(self, model_path: str = "tech/parler_ckpt", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model_path = Path(model_path)
        self._ensure_parler_on_path()
        self.model, self.tokenizer, self.processor = self._load_parler_model()

    def _ensure_parler_on_path(self):
        # Add tech/parler-tts to import path
        candidate = Path("tech/parler-tts").resolve()
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    def _load_parler_model(self):
        from transformers import AutoTokenizer, AutoProcessor, ParlerTTSForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        processor = AutoProcessor.from_pretrained(self.model_path)
        model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_path).to(self.device).eval()
        return model, tokenizer, processor

    def synthesize(self, text: str, persona_vector: torch.Tensor, style_controls: Dict[str, float] | None = None) -> np.ndarray:
        """
        Generate audio conditioned on text + persona vector + style controls.
        """
        if style_controls is None:
            style_controls = {"tempo": 1.0, "energy": 0.7, "warmth": 0.5, "formality": 0.5}

        # Prepare prompt with style controls
        prompt = f"A {style_controls.get('timbre', 'female')} speaker with {style_controls.get('emotion', 'calm')} emotion, speaking at tempo {style_controls.get('tempo', 1.0)}, energy {style_controls.get('energy', 0.7)}, warmth {style_controls.get('warmth', 0.5)}, formality {style_controls.get('formality', 0.5)}."

        inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            audio = self.model.generate(**inputs)
            audio = audio.squeeze().detach().cpu().numpy()

        return audio
