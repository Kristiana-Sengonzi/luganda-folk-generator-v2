import modal

app = modal.App("luganda-folk-generator")

# Direct conversion of your Dockerfile
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Replaces: apt-get install commands
    .apt_install(
        "python3.10",
        "python3-pip", 
        "python3.10-venv",
        "git",
        "ffmpeg",
        "libsndfile1"
    )
    # Replaces: ln -s /usr/bin/python3.10 /usr/bin/python
    .run_commands("ln -sf /usr/bin/python3.10 /usr/bin/python")
   
    # Replaces: pip install torch...
    .pip_install(
        "torch", 
        "torchvision", 
        "torchaudio",
        index_url=["https://download.pytorch.org/whl/cu121"]
    )
    # Replaces: pip install -r requirements.txt
    .pip_install_from_requirements("requirements.txt")
    # Replaces: COPY . . (Modal does this automatically)
)

@app.function(
    image=image,
    gpu="A100",  # Modal provides CUDA automatically
    secrets=[modal.Secret.from_name("bag")],
    timeout=3600  # 1 hour for model downloads
)
@modal.asgi_app()
def fastapi_app():
    """Replaces: CMD ["python", "download_models.py"]"""
    # This runs your download_models.py and starts FastAPI
    from download_models import download_all_models
    download_all_models()
    
    from app.main import app as fastapi_app
    return fastapi_app
