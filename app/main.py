from contextlib import asynccontextmanager
from dotenv import load_dotenv
from app.audio_utils import SmallAudioVAE
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.models_manager import models_manager
from app.audio_utils import denoise_audio, save_audio
from app.audio_vae_utils import generate_continuous_audio
from app.vllm_manager import vllm_manager
from app.lyric_utils import extract_pure_lyrics
from app.lyric_emotion_extractor_utils import LyricEmotionExtractor
from app .audio_utils import SmallAudioVAE
from app.translation_utils import translate_lyrics_to_luganda
import os
import uuid
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modern lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info(" Starting application...")
    logger.info(" Initializing vLLM engine...")
    
    try:
        vllm_manager.initialize_llm()
        logger.info(" vLLM engine initialized!")
    except Exception as e:
        logger.error(f" Failed to initialize vLLM: {str(e)}")
        raise
    
    yield  # App runs here
    
    # Shutdown code (optional)
    logger.info(" Shutting down application...")

# Create FastAPI app with lifespan
app = FastAPI(title="Luganda Folk Song Generator", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/templates"), name="static")
templates = Jinja2Templates(directory="app/templates")



# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    """Health check endpoint"""
    vllm_ready = vllm_manager.llm_engine is not None
    return {
        "status": "ok" if vllm_ready else "starting",
        "vllm_engine_ready": vllm_ready,
        "device": vllm_manager.device
    }



# -------------------------
# Main UI
# -------------------------
@app.get("/", response_class=HTMLResponse)
def get_ui(request: Request):
    themes = [
        "cleverness", "cooperation", "patience", "courage", "wisdom", "integrity",
        "obedience", "caution", "responsibility", "consequences", "respect", "kindness",
        "compassion", "generosity", "gratitude", "friendship", "loyalty", "vigilance",
        "perseverance", "alertness", "empathy", "self control", "contentment",
        "appreciation", "honesty", "humility"
    ]
    return templates.TemplateResponse("index.html", {"request": request, "themes": themes})


# -------------------------
# Generate story + lyrics + audio
# -------------------------
@app.post("/generate")
def generate(
    request: Request,
    theme: str = Form(...),
):
    """Generate story, lyrics, and audio based on theme"""
    try:
        logger.info(f" Generating content for theme: {theme}")
        
        # 1. Generate story
        logger.info(" Generating story...")
        
        
        prompt = (
            f"Theme(s): {theme}\n"
            "Instruction: Write a Buganda folk story with a beginning, middle, and ending "
            "that teaches this moral value. Avoid repeating sentences or phrases."
        )
        
        
       
        story = vllm_manager.generate_story(prompt)
        logger.info(" Story generated!")

        # 2. Generate lyrics (English)
        logger.info("Generating lyrics...")
    
        
        lyric_prompt = f"""Create a complete folksong based on this story. Include [Verse], [Chorus], [Call], and [Response] sections.

        Begin the song with "Start" and end with "End".
        Output ONLY the content between "Start" and "End" - no explanations before or after.

        Story: {story}

        Start"""
        
        
        generated_lyrics = vllm_manager.generate_lyrics(lyric_prompt)
        logger.info(" Lyrics generated!")

        # 3. Process lyrics for emotion and audio generation
        logger.info(" Processing lyrics for audio...")
        clean_lyrics = extract_pure_lyrics(generated_lyrics)
        lyric_extractor = LyricEmotionExtractor()
        processed = lyric_extractor.process_lyrics(clean_lyrics)
        durations, tempos, energies = lyric_extractor.get_audio_arrays(processed)

        # 4. Translate lyrics
        logger.info(" Translating lyrics...")
        translations = translate_lyrics_to_luganda(clean_lyrics)
        lyrics_en = translations["english"]
        lyrics_lg = translations["luganda"]
        logger.info(" Translation complete!")

        # 5. Generate audio using VAE
        logger.info(" Generating audio...")
        
        vae_model = models_manager.load_audio_vae()

        waveform = generate_continuous_audio(
            model=vae_model,
            instrument_idx=0,
            num_segments=len(durations),
            durations=durations,
            tempos=tempos,
            energies=energies,
            latent_dim=32,
            sr=16000,
            device=models_manager.device
        )

        # 6. Save and denoise audio
        logger.info(" Processing audio...")
        audio_filename = f"{uuid.uuid4()}.wav"
        save_audio(waveform, audio_filename)
        clean_audio = denoise_audio(audio_filename, audio_filename.replace(".wav", "_denoised.wav"))
        logger.info(" Audio processing complete!")
        print(f"Looking for audio file: {clean_audio_filename}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")

        try:
            if os.path.exists(clean_audio_filename):
                print(f" Audio file exists: {clean_audio_filename}")
            else:
                print(f" Audio file NOT FOUND: {clean_audio_filename}")

        except Exception as e:
            print(f"Error checking file: {e}")

        logger.info(" Generation complete!")
        
        return {
            "story": story,
            "lyrics_en": lyrics_en,
            "lyrics_lg": lyrics_lg,
            "audio_path": f"/audio/{os.path.basename(clean_audio)}" 
        }
        
    except Exception as e:
        logger.error(f" Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# -------------------------
# Serve audio files
@app.get("/audio/{filename}")
def get_audio(filename: str):
    """Serve generated audio files from current directory"""
    # Files are now saved in current directory, not /tmp/
    audio_path = filename  # Just the filename, since it's in current dir
    
    # Security check: ensure filename is safe
    if '/' in filename or '..' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if os.path.exists(audio_path):
        return FileResponse(audio_path, media_type="audio/wav", filename=filename)
    
    # Also check for files in current directory with the exact name
    raise HTTPException(status_code=404, detail=f"Audio file {filename} not found")
