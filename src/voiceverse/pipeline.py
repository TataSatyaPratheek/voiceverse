# src/voiceverse/pipeline.py
import os
import json
from pathlib import Path
from typing import Dict, List

from prefect import flow, task, get_run_logger

from voiceverse.config import config
from voiceverse.tts import StyleTTSWrapper
from voiceverse.vocoder import BigVGANVocoder, HiFiGANVocoder


def _load_persona_vector(persona_vector_path: str):
    # Accept either .json (vector+config) or .pt (tensor only)
    p = Path(persona_vector_path)
    if p.suffix.lower() == ".json":
        with open(p, "r") as f:
            data = json.load(f)
        import torch
        vec = torch.tensor(data["vector"])
    elif p.suffix.lower() == ".pt":
        import torch
        vec = torch.load(p, map_location="cpu")
    else:
        raise ValueError(f"Unsupported persona vector format: {p.suffix}")
    return vec


@task
def preprocess_dataset(dataset_path: str) -> str:
    logger = get_run_logger()
    logger.info(f"Starting data preprocessing for dataset: {dataset_path}")
    # Add dataset checks or normalization here as needed
    return dataset_path


@task
def fine_tune_model(dataset_path: str, persona_vector_path: str) -> str:
    logger = get_run_logger()
    logger.info(f"Fine tuning model with dataset: {dataset_path} and persona vector: {persona_vector_path}")
    # Hook in your PersonaFineTuner here when ready; for now, just pass through
    return persona_vector_path


@task
def generate_audio(script_path: str, persona_vector_path: str) -> str:
    logger = get_run_logger()
    logger.info(f"Generating audio for script: {script_path} with persona vector: {persona_vector_path}")

    # Load persona vector
    persona_vec = _load_persona_vector(persona_vector_path)

    # Init TTS + Vocoder
    tts = StyleTTSWrapper(checkpoint_path=config.styletts_checkpoint, config_path=config.styletts_config, device="cuda")
    # Prefer BigVGAN for quality; fallback to HiFiGAN config if preferred
    bigv = BigVGANVocoder(checkpoint_path=config.hifigan_checkpoint, device="cuda", use_cuda_kernel=False)

    # Read script, synthesize line by line, concatenate
    script_text = Path(script_path).read_text(encoding="utf-8").strip()
    lines = [ln for ln in script_text.splitlines() if ln.strip()]

    from itertools import chain
    audio_chunks: List[np.ndarray] = []
    import numpy as np

    for ln in lines:
        mel = tts.synthesize(ln, persona_vec, style_controls={
            "tempo": 0.95, "energy": 0.7, "warmth": 0.6, "formality": 0.5
        })
        wav = bigv.vocode(mel, sample_rate=24000)
        audio_chunks.append(wav.astype(np.float32))

    if not audio_chunks:
        raise RuntimeError("No lines in script or synthesis failed")

    # Concatenate with short silences
    sr = 24000
    silence = np.zeros(int(0.2 * sr), dtype=np.float32)
    full = audio_chunks
    for chunk in audio_chunks[1:]:
        full = np.concatenate([full, silence, chunk], axis=0)

    # Write output
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "generated_episode.wav"
    import soundfile as sf
    sf.write(out_fp, full, sr)
    logger.info(f"Wrote {out_fp}")
    return str(out_fp)


@flow(name="VoiceVerse Episode Pipeline")
def episode_pipeline(dataset_path: str, script_path: str, persona_vector_path: str) -> str:
    preprocessed_data = preprocess_dataset(dataset_path)
    updated_persona = fine_tune_model(preprocessed_data, persona_vector_path)
    output_audio = generate_audio(script_path, updated_persona)
    return output_audio


if __name__ == "__main__":
    episode_pipeline(
        dataset_path="data/telugu_english_samples",
        script_path="scripts/episode_001.txt",
        persona_vector_path="personas/telugu_narrator.json"
    )
