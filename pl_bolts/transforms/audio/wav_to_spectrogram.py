import numpy as np

from pl_bolts.utils.warnings import warn_missing_pkg
try:
    import librosa
except ModuleNotFoundError:
    warn_missing_pkg('librosa')  # pragma: no-cover

def spec_to_image(spec, eps=1e-6):
    # https://gist.github.com/hasithsura/b798c972448bed0b4fa0ab891f244d19
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return spec_scaled

def get_melspectrogram_db(
    file_path,
    sr=None,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=8300,
    top_db=80,
):
    # https://gist.github.com/hasithsura/78a1fb909a4271d6287c3b273038177f
    wav, sr = librosa.load(file_path, sr=sr)
    if wav.shape[0] < 5 * sr:
        wav = np.pad(wav, int(np.ceil((5 * sr - wav.shape[0]) / 2)), mode='reflect')
    else:
        wav = wav[: 5 * sr]
    spec = librosa.feature.melspectrogram(
        wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    spec_db = librosa.power_to_db(spec, top_db=top_db)
    return spec_db