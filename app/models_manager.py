import torch
from app.audio_utils import SmallAudioVAE
import torch.nn as nn
import torch.serialization
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import noisereduce as nr
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

            if not os.path.exists(AUDIO_VAE_PATH):
                raise FileNotFoundError(f"Model not found at {AUDIO_VAE_PATH}")

            try:
            #  FIX: Use add_safe_globals with the correct class path
                from app.audio_utils import SmallAudioVAE
            
                with torch.serialization.add_safe_globals([SmallAudioVAE]):
                    self._vae_model = torch.load(AUDIO_VAE_PATH, map_location=self.device)
            
                self._vae_model.eval()
                print(" Model loaded successfully with safe_globals!")
            
            except Exception as e:
                print(f" Model load failed: {e}")
            # Fallback to state dict method
                self._load_as_state_dict()
        
        return self._vae_model

    def _load_as_state_dict(self):
        """Alternative loading method"""
        try:
            from app.audio_utils import SmallAudioVAE
        
            print(" Loading as state dict...")
        
        # Load just the state dict with weights_only=False
            state_dict = torch.load(AUDIO_VAE_PATH, map_location=self.device, weights_only=False)
        
        # Create model instance
            self._vae_model = SmallAudioVAE(
            input_dim=16000*5,
            latent_dim=32, 
            num_instruments=6
            )
        
        # If it's a full model object, extract state dict
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
            self._vae_model.load_state_dict(state_dict)
            self._vae_model.to(self.device)
            self._vae_model.eval()
            print(" State dict load successful!")
        
        except Exception as e:
            print(f" State dict load failed: {e}")
            raise
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
