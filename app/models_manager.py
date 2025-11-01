import torch
import torch.nn as nn
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

class SmallAudioVAE(torch.nn.Module):
    def __init__(self, input_dim=16000*5, latent_dim=32, num_instruments=6):
        super().__init__()
        self.num_instruments = num_instruments
        
        # Encoder: audio + instrument info
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_instruments, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: latent + instrument info
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_instruments, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )

    def encode(self, x, instrument_labels):
        instrument_one_hot = F.one_hot(instrument_labels, num_classes=self.num_instruments).float()
        x_flat = x.view(x.size(0), -1)
        conditioned_input = torch.cat([x_flat, instrument_one_hot], dim=1)
        h = self.encoder(conditioned_input)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, instrument_labels):
        instrument_one_hot = F.one_hot(instrument_labels, num_classes=self.num_instruments).float()
        conditioned_latent = torch.cat([z, instrument_one_hot], dim=1)
        return self.decoder(conditioned_latent)

    def forward(self, x, instrument_labels):
        mu, logvar = self.encode(x, instrument_labels)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z, instrument_labels)
        return reconstructed, mu, logvar

    def generate(self, instrument_labels, z=None):
        if z is None:
            z = torch.randn(instrument_labels.size(0), self.fc_mu.out_features).to(instrument_labels.device)
        generated_audio = self.decode(z, instrument_labels)
        return generated_audio.view(-1, 1, 16000*5)
    

def denoise_audio(input_path, output_path):
    import librosa
    y, sr = librosa.load(input_path, sr=None)
    noise_clip = y[:int(0.5 * sr)]
    reduced = nr.reduce_noise(
        y=y,
        y_noise=noise_clip,
        sr=sr,
        prop_decrease=0.9,
        stationary=False
    )
    sf.write(output_path, reduced, sr)
    return output_path

def save_audio(waveform, filename, sr=16000):
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform)) * 0.9
    sf.write(filename, waveform, sr)
    return filename



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
        
        
        # Create model instance with the same architecture
            self._vae_model = SmallAudioVAE(
                input_dim=16000*5,
                latent_dim=32, 
                num_instruments=6
            )
        
       
            state_dict = torch.load(
                AUDIO_VAE_PATH, 
                map_location=self.device, 
                weights_only=False
            )
        
        
            self._vae_model.load_state_dict(state_dict)
            self._vae_model.to(self.device)
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
