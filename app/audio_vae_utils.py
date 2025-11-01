import numpy as np
import soundfile as sf
import torch
from app.audio_utils import  SmallAudioVAE
from tqdm import tqdm

# =========================================================
# Helper function for smooth concatenation
# =========================================================
def smooth_concatenate(segments, crossfade_duration=0.4, sr=16000):
    """Crossfades between segments for continuous blending"""
    crossfade_samples = int(crossfade_duration * sr)
    output = np.zeros(0)

    for i, seg in enumerate(segments):
        if i == 0:
            output = seg
        else:
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_len = min(crossfade_samples, len(seg), len(output))
            output[-fade_len:] = (
                output[-fade_len:] * fade_out[:fade_len] + seg[:fade_len] * fade_in[:fade_len]
            )
            output = np.concatenate([output, seg[fade_len:]])
    return output


# =========================================================
# Flexible continuous audio generator
# =========================================================
def generate_continuous_audio(
    model,
    instrument_idx=0,
    num_segments=8,
    latent_dim=32,
    sr=16000,
    segment_duration=5.0,
    durations=None,
    tempos=None,
    energies=None,
    device='cuda'
):
    """
    Generate continuous audio, optionally using external arrays (durations, tempos, energies).
    If not provided, random or default values are generated.
    """
    model.eval()
    all_segments = []

    # =========================================================
    # 1️ Handle optional input values
    # =========================================================
    if durations is None:
        durations = np.random.uniform(3, 7, size=num_segments)  # seconds
    if tempos is None:
        tempos = np.random.choice([110, 115, 120, 125, 130, 135], size=num_segments)
    if energies is None:
        energies = np.random.uniform(0.07, 0.10, size=num_segments)

    print(" Using the following parameters:")
    print(f"Durations: {[round(d, 2) for d in durations]}")
    print(f"Tempos: {list(map(str, np.round(tempos, 1)))}")
    print(f"Energies: {list(map(str, np.round(energies, 4)))}\n")

    # Random latent anchors
    latent_points = [torch.randn(1, latent_dim).to(device) for _ in range(num_segments + 1)]
    instrument_labels = torch.tensor([instrument_idx]).to(device)

    print(" Generating smooth latent-interpolated audio...\n")

    with torch.no_grad():
        for i in tqdm(range(num_segments), desc="Generating segments"):
            z_start, z_end = latent_points[i], latent_points[i + 1]
            seg_duration = float(durations[i])
            total_samples = int(seg_duration * sr)

            # Apply the tempo/energy to adjust interpolation density
            tempo_factor = float(tempos[i]) / 120.0
            energy_scale = float(energies[i]) / 0.085

            for alpha in np.linspace(0, 1, int(5 * tempo_factor)):
                z_interp = (1 - alpha) * z_start + alpha * z_end
                generated_audio = model.decode(z_interp, instrument_labels)
                generated_audio = generated_audio.view(-1).cpu().numpy()

                # Scale by energy
                generated_audio *= energy_scale

                # Pad or crop to exact segment length
                generated_audio = np.pad(generated_audio, (0, max(0, total_samples - len(generated_audio))))
                generated_audio = generated_audio[:total_samples]
                all_segments.append(generated_audio)

    # =========================================================
    # 2️ Smoothly concatenate all interpolated segments
    # =========================================================
    final_audio = smooth_concatenate(all_segments, crossfade_duration=0.4, sr=sr)

    # Normalize
    if np.max(np.abs(final_audio)) > 0:
        final_audio = final_audio / np.max(np.abs(final_audio)) * 0.9

    output_filename = f"continuous_latent_mix_inst{instrument_idx}.wav"
    sf.write(output_filename, final_audio, sr)

    print("\n Continuous latent-based audio generated successfully!")
    print(f"   Saved as: {output_filename}")
    print(f"   Duration: {len(final_audio) / sr:.2f}s")
    print(f"   Segments: {num_segments}")
    return final_audio


