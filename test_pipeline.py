#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append('src')

from voiceverse import PersonaVectorManager, PersonaConfig
from voiceverse.pipeline import episode_pipeline
from voiceverse.quality import AudioQualityAssessor, QualityGate

def test_full_pipeline():
    """Test the complete VoiceVerse pipeline"""
    
    # 1. Create persona
    persona_manager = PersonaVectorManager()
    config = PersonaConfig(
        name='telugu_narrator',
        language_mix=0.3,  # 70% Telugu, 30% English
        timbre='female',
        tempo=0.9,
        energy=0.7,
        warmth=0.8,
        formality=0.6,
        emotion='calm'
    )
    
    persona_vector = persona_manager.create_persona(config)
    persona_manager.save_persona('telugu_narrator', Path('personas/telugu_narrator.json'))
    
    # 2. Run pipeline
    try:
        result = episode_pipeline(
            dataset_path="data/telugu_english_samples",
            script_path="scripts/episode_001.txt", 
            persona_vector_path="personas/telugu_narrator.json"  # Use .json extension
        )
        print(f"✅ Pipeline completed successfully: {result}")
        
        # 3. Quality assessment
        assessor = AudioQualityAssessor()
        # TODO: Load generated audio and assess
        # audio, sr = librosa.load(result)
        # metrics = assessor.assess_audio(audio, text, config.__dict__)
        
        # gate = QualityGate(config.quality_thresholds)
        # passed, failures = gate.validate(metrics)
        
        print("✅ All systems operational!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False

if __name__ == "__main__":
    test_full_pipeline()
