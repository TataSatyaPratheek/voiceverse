# src/voiceverse/tts.py
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import re
from voiceverse.cross_import import import_symbol_from_file

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
    """StyleTTS2 wrapper with LJSpeech and auxiliary models loaded as in notebook."""
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.config_path = Path(config_path).resolve()
        self.model, self.sampler = self._load_styletts2_model()

    def _load_styletts2_model(self):
        import yaml
        import builtins
        import torch
        
        torch.serialization.add_safe_globals([builtins.getattr])
        st_root = Path("tech/StyleTTS2").resolve()
        sys.path.insert(0, str(st_root))
        
        # Load config
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        from munch import Munch
        def recursive_munch(x):  # helper for nested config objects
            return Munch({k: recursive_munch(v) if isinstance(v, dict) else v for k, v in x.items()}) if isinstance(x, dict) else x

        model_params = recursive_munch(config.get("model_params", {}))
        
        # Load auxiliary models
        asr_weights_path = Path(config.get("ASR_path") or config.get("asr_weights", "Utils/ASR/epoch_00080.pth"))
        asr_config_path = Path(config.get("ASR_config") or config.get("asr_config", "Utils/ASR/config.yml"))
        f0_path = Path(config.get("F0_path") or config.get("f0_path", "Utils/JDC/bst.t7"))
        plbert_dir = Path(config.get("PLBERT_dir", "Utils/PLBERT"))

        # Resolve all paths to StyleTTS2 root
        asr_weights_path = st_root / asr_weights_path if not asr_weights_path.is_absolute() else asr_weights_path
        asr_config_path = st_root / asr_config_path if not asr_config_path.is_absolute() else asr_config_path
        f0_path = st_root / f0_path if not f0_path.is_absolute() else f0_path
        plbert_dir = Path(config.get("PLBERT_dir", "Utils/PLBERT"))
        if not plbert_dir.is_absolute():
            plbert_dir = st_root / plbert_dir

        # Import helpers from StyleTTS2 repo
        styletts_models_py = Path("/home/vi/Documents/audio_startup/tech/StyleTTS2/models.py")
        plbert_util_py = Path("/home/vi/Documents/audio_startup/tech/StyleTTS2/Utils/PLBERT/util.py")
        
        load_ASR_models = import_symbol_from_file(styletts_models_py, "load_ASR_models")
        load_F0_models = import_symbol_from_file(styletts_models_py, "load_F0_models")
        build_model = import_symbol_from_file(styletts_models_py, "build_model")

        load_plbert = import_symbol_from_file(plbert_util_py, "load_plbert")

        asr_model = load_ASR_models(str(asr_weights_path), str(asr_config_path))
        pitch_extractor = load_F0_models(str(f0_path))
        plbert = load_plbert(str(plbert_dir))

        # Build model as in notebook's build_model call
        model = build_model(model_params, asr_model, pitch_extractor, plbert)

        # Load weights (note: some checkpoints use 'net', 'model', or whole dict)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        params = ckpt.get("net") or ckpt.get("model") or ckpt
        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key])
                except Exception:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith("module.") else k
                        new_state_dict[name] = v
                    model[key].load_state_dict(new_state_dict, strict=False)
        for k in model: model[k].eval(), model[k].to(self.device)

        # Load sampler as in notebook
        try:
            diffusion_sampler_py = Path("/home/vi/Documents/audio_startup/tech/StyleTTS2/Modules/diffusion/sampler.py")
            DiffusionSampler = import_symbol_from_file(diffusion_sampler_py, "DiffusionSampler")
            ADPM2Sampler = import_symbol_from_file(diffusion_sampler_py, "ADPM2Sampler")
            KarrasSchedule = import_symbol_from_file(diffusion_sampler_py, "KarrasSchedule")
            sampler = DiffusionSampler(
                model['diffusion']['diffusion'],
                sampler=ADPM2Sampler(),
                sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                clamp=False
            )
        except Exception:
            sampler = None

        return (model, sampler)

    def synthesize(self, text: str, persona_vector: torch.Tensor, style_controls: Dict[str, float] | None = None) -> np.ndarray:
        """
        Generate mel-spectrogram using the diffusion sampler.
        """
        if style_controls is None:
            style_controls = {"tempo": 1.0, "energy": 0.7, "warmth": 0.5, "formality": 0.5}

        # Preprocess text
        tokens = self._preprocess_multilingual_text(text)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(self.device)

        # Persona vector
        persona = persona_vector.to(self.device).float().unsqueeze(0)

        # Use sampler for inference
        with torch.no_grad():
            if self.sampler:
                mel = self.sampler(tokens, persona, style_controls)
                mel = mel.squeeze(0).detach().cpu().float().numpy()
            else:
                # Fallback if sampler not available
                mel = self.model['diffusion'].infer(tokens, persona, **style_controls)
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
