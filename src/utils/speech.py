from __future__ import annotations

# Standard libraries
from typing import Tuple

# Third-party libraries
import torch


def load_audio_mono_16k(path: str) -> Tuple[torch.Tensor, int]:
    """
    Returns: (waveform [T] float32), sample_rate (int)
    Uses torchaudio if available, otherwise falls back to soundfile.
    """
    try:
        import torchaudio

        wav, sr = torchaudio.load(path)

        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=False)
        elif wav.dim() == 2:
            wav = wav.squeeze(0)

        wav = wav.to(torch.float32)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        return wav, sr

    except Exception:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=False)

        if wav.ndim == 2:
            wav = wav.mean(axis=1).astype("float32")

        if sr != 16000:
            from scipy.signal import resample_poly

            gcd = __import__("math").gcd(sr, 16000)
            up = 16000 // gcd
            down = sr // gcd
            wav = resample_poly(wav, up, down).astype("float32")
            sr = 16000

        return torch.from_numpy(wav), sr
