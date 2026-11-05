import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "samples/listen"
SPEC = BASE / "spectrograms"
SPEC.mkdir(exist_ok=True)


def load48(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 48000:
        g = np.gcd(48000, sr)
        x = resample_poly(x, 48000 // g, sr // g)
    return x.astype(np.float32)


def plot(src, dst, title, seconds=None):
    x = load48(BASE / src)
    if seconds is not None:
        x = x[: int(seconds * 48000)]
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=2048, hop_length=480)), ref=np.max)
    plt.figure(figsize=(6.6, 3.6))
    librosa.display.specshow(S, sr=48000, hop_length=480, x_axis="time", y_axis="linear",
                             cmap="magma", vmin=-80, vmax=0, fmin=0, fmax=24000)
    plt.ylim(0, 24000)
    plt.yticks(np.arange(0, 24001, 5000), ["0", "5", "10", "15", "20"])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("kHz")
    plt.tight_layout()
    plt.savefig(SPEC / dst, dpi=150)
    plt.close()


en = ["OSR_us_000_0010", "OSR_us_000_0013", "OSR_us_000_0057", "OSR_us_000_0059"]
zh = ["OSR_cn_000_0072", "OSR_cn_000_0073", "OSR_cn_000_0074", "OSR_cn_000_0075"]
for n in en + zh:
    plot(f"8k/{n}_input_8k.wav", f"{n}_input_8k_spec.png", "8k input", 10)
    plot(f"48k_from8k/{n}_output_48k.wav", f"{n}_output_48k_spec.png", "8k -> 48k", 10)
plot("8k/samplelib_speech_1m_input_8k.wav", "samplelib_speech_1m_input_8k_spec.png", "8k input", 10)
plot("48k_from8k/samplelib_speech_1m_output_48k.wav", "samplelib_speech_1m_output_48k_spec.png", "8k -> 48k", 10)
for n in ["speech_2", "speech_3"]:
    plot(f"8k/{n}_input_8k.wav", f"{n}_input_8k_spec.png", "8k input")
    plot(f"48k_from8k/{n}_output_48k.wav", f"{n}_output_8k_48k_spec.png", "8k -> 48k")
    plot(f"16k/{n}_input_16k.wav", f"{n}_input_16k_spec.png", "16k input")
    plot(f"48k_from16k/{n}_output_48k.wav", f"{n}_output_16k_48k_spec.png", "16k -> 48k")
print("images", len(list(SPEC.glob("*.png"))))
