import modal
# IMPORTANT: This line is required to reference the download function during the image build
import download_models 

app = modal.App("luganda-folk-generator")

# 1. Define the Image/Environment
# Using modal.Image is cleaner and faster than manual Dockerfile steps.
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install necessary OS packages (like ffmpeg for audio)
    # We remove the redundant python3/pip apt installs.
    .apt_install(
        "git",
        "ffmpeg", 
        "libsndfile1"
    )
    # Correctly installing PyTorch and related packages.
    # The --index-url is passed via extra_options to avoid the InvalidError.
    .pip_install(
        "torch", 
        "torchvision", 
        "torchaudio",
        index_url:"https://download.pytorch.org/whl/cu121"
    )
    # Install dependencies from your local requirements file
    .pip_install_from_requirements("requirements.txt")
    
    # 🌟 CRITICAL FIX: Run the model download function during the image build.
    # This "bakes" the large model into the image, eliminating slow cold starts.
    .run_function(download_models.download_all_models)
)

# 2. Define the Function/App
@app.function(
    image=image,
    gpu="A100", # Modal handles CUDA drivers automatically
    secrets=[modal.Secret.from_name("bag")], # Uses your correctly named secret
    timeout=3600, # 1 hour timeout for long running tasks
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def fastapi_app():
    """This function starts your FastAPI web service."""
    # Since download_all_models ran during the image build, the model is ready!
    
    from app.main import app as fastapi_app
    return fastapi_app
