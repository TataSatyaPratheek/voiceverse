from prefect import flow
from voiceverse.pipeline import generate_episode

@flow
def run_episode(script_path: str, persona: str):
    return generate_episode(script_path, persona)
