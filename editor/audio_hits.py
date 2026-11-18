import subprocess
import numpy as np
from scipy import signal


def has_audio(video_path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", video_path],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def load_audio(video_path, sample_rate=22050):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", video_path, "-vn", "-ac", "1",
                          "-ar", str(sample_rate), "-f", "f32le", "-"],
                         capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32), sample_rate


def detect_hit_times(video_path, band=(1000, 4000), smooth_ms=10, min_gap_s=0.25,
                     prominence_ratio=6.0):
    """Racket/bounce impacts are short broadband transients. Band-pass, take the
    envelope, and keep peaks that stand well above the local noise floor (median-based,
    so steady crowd/commentary noise raises the bar instead of creating hits)."""
    if not has_audio(video_path):
        return []
    audio, sr = load_audio(video_path)
    if audio.size == 0:
        return []

    sos = signal.butter(4, band, btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, audio)

    hop = int(sr * smooth_ms / 1000)
    n = len(filtered) // hop
    envelope = np.abs(filtered[:n * hop]).reshape(n, hop).max(axis=1)
    env_rate = sr / hop

    # rolling noise floor over ~2 s
    floor = signal.medfilt(envelope, kernel_size=int(env_rate * 2) | 1)
    floor = np.maximum(floor, 1e-6)
    peaks, _ = signal.find_peaks(envelope / floor, height=prominence_ratio,
                                 distance=max(1, int(min_gap_s * env_rate)))
    return (peaks / env_rate).tolist()
