import modal
from download_models import download_all_models
app = modal.App("luganda-folk-generator")

# 1. Define the Image/Environment
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "ffmpeg", 
        "libsndfile1"
    )
    .pip_install(
        "torch", 
        "torchvision", 
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install_from_requirements("requirements.txt")
    
    # 🌟 CRITICAL: Run the model download during image build WITH secrets
    .run_function(
        download_all_models,
        secrets=[modal.Secret.from_name("bag2")]
)   
    .add_local_dir(".", "/root")
)

# 2. Define the Function/App
@app.function(
    image=image,
    gpu="A100",
    secrets=[modal.Secret.from_name("bag2")],
    timeout=3600
)


@modal.asgi_app()
def fastapi_app():
    """ASGI app for web serving"""
    from app.main import app as fastapi_app
    return fastapi_app
