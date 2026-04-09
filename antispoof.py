"""
Step 3: Anti-Spoof Layer
Detects photo/video/fake face attacks using multiple signals:
- Frequency analysis (FFT)
- Color distribution (YCbCr skin model)
- Moiré pattern detection (screen replay)
- Edge density analysis (Laplacian)
"""

import cv2
import numpy as np


def frequency_analysis(face: np.ndarray) -> float:
    """Real faces have richer high-frequency content than printed/screen images."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
    gray = cv2.resize(gray, (128, 128))

    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4

    y, x = np.ogrid[:h, :w]
    low_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    high_mask = ~low_mask

    low_energy = np.sum(magnitude[low_mask])
    high_energy = np.sum(magnitude[high_mask])
    total = low_energy + high_energy

    if total == 0:
        return 0.0

    ratio = high_energy / total
    return float(np.clip((ratio - 0.3) / 0.4, 0.0, 1.0))


def color_distribution_analysis(face: np.ndarray) -> float:
    """Real skin has characteristic Cb/Cr distributions in YCbCr space."""
    if len(face.shape) != 3:
        return 0.5

    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].flatten()
    cb = ycrcb[:, :, 2].flatten()

    cr_in_range = np.mean((cr >= 130) & (cr <= 180))
    cb_in_range = np.mean((cb >= 75) & (cb <= 130))
    range_score = (cr_in_range + cb_in_range) / 2

    cr_std = np.std(cr)
    cb_std = np.std(cb)
    variance_score = np.clip(min(cr_std, cb_std) / 15.0, 0.0, 1.0)

    return float(0.6 * range_score + 0.4 * variance_score)


def moire_pattern_detection(face: np.ndarray) -> float:
    """Detect moiré patterns common in screen replay attacks.
    Low energy = no moiré = real face → high score."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
    gray = cv2.resize(gray, (128, 128))

    blur_low = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_high = cv2.GaussianBlur(gray, (15, 15), 0)
    bandpass = cv2.subtract(blur_low, blur_high)

    energy = np.mean(np.abs(bandpass.astype(np.float32)))
    return float(np.clip(1.0 - (energy / 30.0), 0.0, 1.0))


def edge_density_analysis(face: np.ndarray) -> float:
    """Real faces have moderate edge density (not too blurry, not too sharp)."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
    gray = cv2.resize(gray, (128, 128))

    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    if variance < 20:
        return 0.2
    elif variance > 1000:
        return 0.3
    else:
        return float(np.clip((variance - 20) / 200, 0.3, 1.0))


def compute_antispoof_score(face: np.ndarray) -> dict:
    """
    Compute combined anti-spoof score from all signals.
    Returns: {score, is_real, breakdown}
    """
    freq = frequency_analysis(face)
    color = color_distribution_analysis(face)
    moire = moire_pattern_detection(face)
    edge = edge_density_analysis(face)

    combined = 0.30 * freq + 0.25 * color + 0.25 * moire + 0.20 * edge

    return {
        "score": round(combined, 4),
        "is_real": combined > 0.45,
        "breakdown": {
            "frequency": round(freq, 4),
            "color": round(color, 4),
            "moire": round(moire, 4),
            "edge": round(edge, 4),
        },
    }
