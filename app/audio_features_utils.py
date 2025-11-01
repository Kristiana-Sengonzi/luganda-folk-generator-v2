
import json
import os

# Path to your emotion profile JSON (mounted in container)

EMOTION_JSON_PATH = "app/classical_emotion_features.json"

# Load data once at startup
if not os.path.exists(EMOTION_JSON_PATH):
    raise FileNotFoundError(f"Emotion profile file not found at {EMOTION_JSON_PATH}")

with open(EMOTION_JSON_PATH, "r") as f:
    emotion_data = json.load(f)

def get_audio_features(emotion: str):
    """
    Fetch average audio feature profile (tempo, energy) for a given emotion.
    """
    profiles = emotion_data.get("emotion_profiles", {})
    
    if emotion not in profiles:
        available = list(profiles.keys())
        raise ValueError(f"Emotion '{emotion}' not found. Available: {available}")

    profile = profiles[emotion]
    return {
        "tempo": profile.get("avg_tempo"),
        "rms_energy": profile.get("avg_energy"),
    }
