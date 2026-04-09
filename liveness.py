"""
Step 5: Liveness Layer — Motion + Texture Analysis
Combines frame-to-frame motion detection with LBP texture analysis.
"""

import cv2
import numpy as np


def compute_motion_score(frames: list) -> float:
    """
    Compute motion score from consecutive frames.
    Also checks motion variance (real movement is irregular, replay is smooth).
    Returns: normalized score 0-1
    """
    if len(frames) < 2:
        return 0.0

    gray_frames = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
        g = cv2.resize(g, (160, 160))
        gray_frames.append(g)

    motion_values = []
    for i in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i - 1])
        motion_values.append(np.mean(diff))

    avg_motion = np.mean(motion_values) if motion_values else 0.0
    motion_var = np.var(motion_values) if len(motion_values) > 1 else 0.0

    motion_norm = np.clip(avg_motion / 20.0, 0.0, 1.0)
    variance_norm = np.clip(motion_var / 10.0, 0.0, 1.0)

    return float(0.6 * motion_norm + 0.4 * variance_norm)


def compute_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Compute Local Binary Pattern histogram for texture analysis.
    LBP captures micro-texture patterns: real skin vs flat printed surfaces.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.resize(gray, (64, 64))  # Smaller for speed

    h, w = gray.shape
    lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = gray[i, j]
            binary = 0
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                y = int(round(i + radius * np.sin(angle)))
                x = int(round(j + radius * np.cos(angle)))
                y = max(0, min(y, h - 1))
                x = max(0, min(x, w - 1))
                if gray[y, x] >= center:
                    binary |= 1 << k
            lbp[i - radius, j - radius] = binary

    hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
    return hist


def compute_texture_score(face: np.ndarray) -> float:
    """
    Texture authenticity score via LBP.
    Real faces have higher entropy (more varied textures).
    Returns: score 0-1
    """
    lbp_hist = compute_lbp(face)

    # Entropy: real faces have higher LBP entropy
    nonzero = lbp_hist[lbp_hist > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))

    # Uniformity: printed images are more uniform
    uniformity = np.sum(lbp_hist ** 2)

    entropy_score = np.clip((entropy - 3.0) / 4.0, 0.0, 1.0)
    uniformity_score = np.clip(1.0 - uniformity * 5, 0.0, 1.0)

    return float(0.6 * entropy_score + 0.4 * uniformity_score)


def compute_liveness_score(frames: list, face_crop: np.ndarray = None) -> dict:
    """
    Combined liveness = 50% motion + 50% texture.
    Returns: {score, is_live, motion_score, texture_score}
    """
    motion_score = compute_motion_score(frames)

    texture_source = face_crop if face_crop is not None else frames[0]
    texture_score = compute_texture_score(texture_source)

    combined = 0.5 * motion_score + 0.5 * texture_score

    return {
        "score": round(combined, 4),
        "is_live": combined > 0.40,
        "motion_score": round(motion_score, 4),
        "texture_score": round(texture_score, 4),
    }
