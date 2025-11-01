import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import noisereduce as nr

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
