"""
Step 4: Realtime Challenge — Blink / Head Turn Detection
Uses MediaPipe Face Mesh for landmark-based action verification.
Falls back to motion-based check if MediaPipe is unavailable.
"""

import cv2
import numpy as np
import random
import uuid

# Active challenge sessions
_sessions = {}
CHALLENGE_TYPES = ["blink", "turn_left", "turn_right", "nod"]


def generate_challenge() -> dict:
    """Generate a random challenge for the user."""
    session_id = str(uuid.uuid4())
    action = random.choice(CHALLENGE_TYPES)
    _sessions[session_id] = {"action": action, "verified": False}
    return {"session_id": session_id, "action": action}


def _get_ear(landmarks, h, w):
    """Compute Eye Aspect Ratio from Face Mesh landmarks."""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    l1 = np.linalg.norm(pt(160) - pt(144))
    l2 = np.linalg.norm(pt(158) - pt(153))
    l3 = np.linalg.norm(pt(33) - pt(133))
    left = (l1 + l2) / (2.0 * l3) if l3 > 0 else 0

    r1 = np.linalg.norm(pt(385) - pt(380))
    r2 = np.linalg.norm(pt(387) - pt(373))
    r3 = np.linalg.norm(pt(362) - pt(263))
    right = (r1 + r2) / (2.0 * r3) if r3 > 0 else 0

    return (left + right) / 2.0


def _get_head_pose(landmarks, h, w):
    """Estimate yaw/pitch from key facial landmarks."""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    nose = pt(1)
    left_eye, right_eye, chin = pt(33), pt(263), pt(152)

    ld = np.linalg.norm(nose - left_eye)
    rd = np.linalg.norm(nose - right_eye)
    total = ld + rd
    yaw = (rd - ld) / total if total > 0 else 0

    eye_cy = (left_eye[1] + right_eye[1]) / 2
    fh = chin[1] - eye_cy
    pitch = ((nose[1] - eye_cy) / fh - 0.5) if fh > 0 else 0

    return {"yaw": float(yaw), "pitch": float(pitch)}


def verify_challenge(frames: list, action: str, session_id: str = None) -> dict:
    """
    Verify that the user performed the requested challenge.
    Returns: {passed, confidence, details}
    """
    try:
        import mediapipe as mp
    except ImportError:
        return _fallback_verify(frames, action)

    mp_mesh = mp.solutions.face_mesh
    ears, yaws, pitches = [], [], []

    with mp_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    ) as mesh:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lms = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            ears.append(_get_ear(lms, h, w))
            pose = _get_head_pose(lms, h, w)
            yaws.append(pose["yaw"])
            pitches.append(pose["pitch"])

    if len(ears) < 3:
        return {"passed": False, "confidence": 0.0, "details": "Insufficient landmarks"}

    if action == "blink":
        ear_range = max(ears) - min(ears)
        passed = ear_range > 0.05 and min(ears) < 0.22
        conf = min(ear_range / 0.1, 1.0)
        detail = f"EAR range={ear_range:.3f}"
    elif action == "turn_left":
        yr = max(yaws) - min(yaws)
        passed = max(yaws) > 0.08 and yr > 0.06
        conf = min(yr / 0.15, 1.0)
        detail = f"Yaw range={yr:.3f}"
    elif action == "turn_right":
        yr = max(yaws) - min(yaws)
        passed = min(yaws) < -0.08 and yr > 0.06
        conf = min(yr / 0.15, 1.0)
        detail = f"Yaw range={yr:.3f}"
    elif action == "nod":
        pr = max(pitches) - min(pitches)
        passed = pr > 0.05
        conf = min(pr / 0.1, 1.0)
        detail = f"Pitch range={pr:.3f}"
    else:
        passed, conf, detail = False, 0.0, f"Unknown: {action}"

    if session_id and session_id in _sessions:
        _sessions[session_id]["verified"] = passed

    return {"passed": passed, "confidence": round(float(conf), 4), "details": detail}


def _fallback_verify(frames, action):
    """Basic motion-based fallback when MediaPipe is unavailable."""
    if len(frames) < 2:
        return {"passed": False, "confidence": 0.0, "details": "Not enough frames"}

    diffs = []
    for i in range(1, len(frames)):
        g1 = cv2.resize(cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY), (160, 160))
        g2 = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), (160, 160))
        diffs.append(np.mean(cv2.absdiff(g1, g2)))

    avg = np.mean(diffs)
    passed = avg > 3.0 and max(diffs) > 5.0
    return {
        "passed": passed,
        "confidence": round(min(avg / 10.0, 1.0), 4),
        "details": f"Fallback motion avg={avg:.2f}",
    }


def cleanup_session(session_id: str):
    """Remove a completed challenge session."""
    _sessions.pop(session_id, None)
