
import os
import subprocess
import sys


def download_all_models():
    from huggingface_hub import snapshot_download
    """Download ALL models from Hugging Face"""
    print("🚀 Starting model downloads...")
    
    models_to_download = {
        "base_model": "krisseng/luganda-llama-3.2-3b",
        "story_lora": "krisseng/story-lora", 
        "lyric_lora": "krisseng/lyric-lora",
        "translator": "krisseng/luganda-translator",
        "audio_vae": "krisseng/audio-vae",
        "emotion": "krisseng/emotion-classifier"
    }

    for model_name, repo_id in models_to_download.items():
        print(f" Downloading {model_name}...")
        
        local_dir = f"/folkstory-app/models/{model_name}"
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=os.getenv("HUGGINGFACE_HUB_TOKEN")
        )
        print(f" {model_name} downloaded!")

    print(" All models downloaded! Starting FastAPI server...")
    return True

if __name__ == "__main__":
    success = download_all_models()
    if success:
        subprocess.run([
            "uvicorn", "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--workers", "1"
        ])
    else:
        sys.exit(1)
