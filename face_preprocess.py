"""
Step 2: Face Detection & Preprocessing using RetinaFace
Detects faces, crops with padding, validates presence before pipeline.
"""

import cv2
import numpy as np
from deepface import DeepFace


def detect_face(frame: np.ndarray, min_confidence: float = 0.90) -> dict | None:
    """
    Detect and crop face from a frame using RetinaFace.
    Returns dict with face_crop, region, confidence — or None if no face found.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="retinaface",   
            enforce_detection=False,
            align=True
        )

        if not faces:
            return None

        best = max(faces, key=lambda f: f.get("confidence", 0))

        if best["confidence"] < min_confidence:
            return None

        region = best["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        # Add 20% padding for context
        pad_w, pad_h = int(w * 0.2), int(h * 0.2)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)

        face_crop = frame[y1:y2, x1:x2]

        return {
            "face_crop": face_crop,
            "region": {"x": x, "y": y, "w": w, "h": h},
            "confidence": best["confidence"],
        }

    except Exception:
        return None
