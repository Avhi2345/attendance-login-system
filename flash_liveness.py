"""
Active Flash Liveness (Web-based 3D Reflection)
Verifies that the face actively reflects ambient colored light (Red, Green, Blue)
emitted by the screen during the challenge phase. 
Deepfakes and 2D photos fail to simulate proper channel lighting shifts in real-time.
"""

import cv2
import numpy as np

def compute_flash_liveness(frames: list, sequence: list) -> dict:
    """
    Analyzes the color channels of sequential frames corresponding to screen color flashes.
    frames: List of BGR numpy arrays (OpenCV default)
    sequence: List of colors flashed, e.g. ["red", "green", "blue"]
    """
    if len(frames) != len(sequence):
        return {"passed": False, "confidence": 0.0, "details": "Frame count mismatch"}

    channel_means = []
    
    for f in frames:
        # Extract face center region to minimize background interference
        h, w = f.shape[:2]
        pad_y = int(h * 0.25)
        pad_x = int(w * 0.25)
        center_face = f[pad_y:h-pad_y, pad_x:w-pad_x]
        
        # OpenCV uses BGR format
        b_mean = np.mean(center_face[:, :, 0])
        g_mean = np.mean(center_face[:, :, 1])
        r_mean = np.mean(center_face[:, :, 2])
        
        channel_means.append({"r": r_mean, "g": g_mean, "b": b_mean})

    # Verify that the expected channel spiked during its corresponding flash
    passed = True
    confidence_sum = 0
    
    for i, expected_color in enumerate(sequence):
        means = channel_means[i]
        
        if expected_color == "red":
            # Red channel should be dominant
            if means["r"] <= means["g"] and means["r"] <= means["b"]:
                passed = False
            confidence_sum += (means["r"] - max(means["g"], means["b"]))
            
        elif expected_color == "green":
            # Green channel should be dominant
            if means["g"] <= means["r"] and means["g"] <= means["b"]:
                passed = False
            confidence_sum += (means["g"] - max(means["r"], means["b"]))
            
        elif expected_color == "blue":
            # Blue channel should be dominant
            if means["b"] <= means["r"] and means["b"] <= means["g"]:
                passed = False
            confidence_sum += (means["b"] - max(means["r"], means["g"]))
            
        else:
            # Baseline / black / white
            pass

    # Basic normalization for confidence
    conf = min(max(confidence_sum / (len(sequence) * 5.0), 0.0), 1.0)
    
    if not passed:
        conf = 0.0

    return {
        "passed": passed, 
        "confidence": round(float(conf), 4), 
        "details": f"Channel analysis matched sequence: {passed}"
    }
