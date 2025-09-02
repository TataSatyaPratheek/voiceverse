from prefect import flow
from .pipeline import episode_pipeline

@flow
def run_episode(dataset_path: str, script_path: str, persona_vector_path: str):
    """Prefect flow for end-to-end episode generation"""
    return episode_pipeline(dataset_path, script_path, persona_vector_path)

if __name__ == "__main__":
    result = run_episode(
        dataset_path="data/telugu_english_samples",  
        script_path="scripts/episode_001.txt",
        persona_vector_path="personas/telugu_narrator.pt"
    )
    print(f"Generated episode: {result}")
