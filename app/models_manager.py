import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from app.config import STORY_GENERATOR_PATH,LYRIC_GENERATOR_PATH,TRANSLATOR_PATH,AUDIO_VAE_PATH,BASE_MODEL_PATH,HARTMANN_EMOTION_PATH
from peft import PeftModel, PeftConfig
from huggingface_hub import login
from app.audio_vae_utils import generate_continuous_audio
import os
from dotenv import load_dotenv



class ModelsManager:
    def __init__(self, device="cpu"):
        self.device = device
        
        # Other models
        self._translator_pipeline = None
        self._vae_model = None
        self._hartmann_pipeline = None
        self._hf_logged_in = False

   

    # -------------------------
    # Load base model ONCE
    # -------------------------
    
    # -------------------------
    def load_translator(self):
        if self._translator_pipeline is None:
            print(" Loading translator...")
            self._translator_tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_PATH)
            self._translator_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR_PATH)
            self._translator_pipeline = pipeline(
                "translation",
                model=self._translator_model,
                tokenizer=self._translator_tokenizer
            )
            print(" Translator loaded!")
        return self._translator_pipeline

    # -------------------------
    # Audio VAE
    # -------------------------
    def load_audio_vae(self):
        if self._vae_model is None:
            print(" Loading audio VAE...")
            self._vae_model = torch.load(
                AUDIO_VAE_PATH, 
                map_location=self.device, 
                weights_only=False
            )
            self._vae_model.eval()
            print(" Audio VAE loaded!")
        return self._vae_model

    # -------------------------
    # Hartmann emotion classifier
    # -------------------------
    def load_hartmann_model(self):
        if self._hartmann_pipeline is None:
            print(" Loading Hartmann emotion classifier...")
            self._hartmann_pipeline = pipeline(
                "text-classification",
                model=HARTMANN_EMOTION_PATH,
                device=-1  # CPU
            )
            print(" Hartmann model loaded!")
        return self._hartmann_pipeline

    # -------------------------
    # Generate audio from emotions
    # -------------------------
    def generate_audio_from_emotions(self, durations, tempos, energies):
        waveform = generate_continuous_audio(
            self.load_audio_vae(),
            instrument_idx=0,
            num_segments=len(durations),
            durations=durations,
            tempos=tempos,
            energies=energies,
            latent_dim=32,
            sr=16000,
            device=self.device
        )
        return waveform

# Singleton instance
models_manager = ModelsManager(device="cuda" if torch.cuda.is_available() else "cpu")