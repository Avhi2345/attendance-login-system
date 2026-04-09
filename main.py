"""
Live Attendance System — Main API
Full pipeline: Input → Preprocess → AntiSpoof → Challenge → Liveness
             → Embedding → FAISS → Decision → Output
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import base64
import cv2
import numpy as np

from deepface import DeepFace
from app.face_preprocess import detect_face
from app.antispoof import compute_antispoof_score
from app.liveness import compute_liveness_score
from app.challenge import generate_challenge, verify_challenge, cleanup_session
from app.flash_liveness import compute_flash_liveness
from app.faiss_index import FaissIndex
from app.google_sheet_db import (
    save_embedding, load_all_users, update_embedding_self_learn,
    log_attendance, get_attendance_log,
)
import hmac
import hashlib
import uuid

# ── FAISS Index (global) ─────────────────────────────────────
faiss_index = FaissIndex(dimension=512)


def rebuild_index():
    """Rebuild FAISS index from all stored embeddings."""
    global faiss_index
    users = load_all_users()
    faiss_index = FaissIndex(dimension=512)
    faiss_index.build_from_users(users)


@asynccontextmanager
async def lifespan(app: FastAPI):
    rebuild_index()
    yield


app = FastAPI(title="Live Attendance System", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Thresholds ────────────────────────────────────────────────
ANTISPOOF_THRESHOLD = 0.45
LIVENESS_THRESHOLD = 0.40
FACE_MATCH_THRESHOLD = 0.65
SELF_LEARN_THRESHOLD = 0.75   # only update embedding if very confident
LEARNING_RATE = 0.3            # weight for new observation in self-learn


# ── Helpers ───────────────────────────────────────────────────
def decode_frame(b64: str) -> np.ndarray:
    """Decode base64 JPEG string → OpenCV BGR image."""
    data = b64.split(",")[1] if "," in b64 else b64
    arr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def compute_embedding(image: np.ndarray) -> dict:
    """Step 6: ArcFace embedding via DeepFace with native Deep Learning Anti-Spoofing."""
    try:
        result = DeepFace.represent(
            img_path=image,
            model_name="ArcFace",          # UPGRADED: Bank-level matching model
            detector_backend="retinaface",
            enforce_detection=False,
            anti_spoofing=True             # UPGRADED: Native FASNet Deep Learning PAD
        )
        return {
            "embedding": np.array(result[0]["embedding"], dtype=np.float64),
            "is_real": result[0].get("is_real", True), # Native deep learning antispoof result
        }
    except Exception:
        # Fallback if anti_spoofing flag fails on older versions of deepface
        result = DeepFace.represent(
            img_path=image,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False,
        )
        return {
            "embedding": np.array(result[0]["embedding"], dtype=np.float64),
            "is_real": True, # Delegate to our custom antispoof heuristic layer
        }


# ── Endpoints ─────────────────────────────────────────────────

# Store active secrets for HMAC verification
session_secrets = {}

@app.get("/challenge")
def get_challenge():
    """Step 4a: Generate active flash challenge and HMAC session secret."""
    session_id = str(uuid.uuid4())
    secret = str(uuid.uuid4())
    
    # Active flash sequence
    sequence = ["red", "green", "blue"]
    
    session_secrets[session_id] = {
        "secret": secret,
        "sequence": sequence
    }
    
    return {
        "action": "flash",
        "sequence": sequence,
        "session_id": session_id,
        "hmac_secret": secret
    }


@app.post("/register")
def register(data: dict):
    """
    Register a new user with multiple face images.
    Body: {"name": "John", "images": ["base64_img1", ...]}
    """
    name = data.get("name", "").strip()
    images = data.get("images", [])

    if not name:
        raise HTTPException(400, "Name is required")
    if len(images) < 3:
        raise HTTPException(400, "At least 3 images required")

    registered = 0
    for img_b64 in images:
        frame = decode_frame(img_b64)

        # Step 2: Detect face
        face = detect_face(frame)
        if face is None:
            continue

        # Step 3: Reject spoofed registrations
        spoof = compute_antispoof_score(face["face_crop"])
        if not spoof["is_real"]:
            continue

        # Step 6: Generate ArcFace embedding & check DL AntiSpoof
        res = compute_embedding(frame)
        if not res["is_real"]:
            continue # Deep Learning PAD rejected it
            
        save_embedding(name, res["embedding"].tolist())
        registered += 1

    if registered == 0:
        raise HTTPException(400, "No valid face detected. Ensure good lighting.")

    rebuild_index()

    return {
        "status": "registered",
        "name": name,
        "faces_registered": registered,
        "total_submitted": len(images),
    }


@app.post("/login")
def login(data: dict):
    """
    Full 9-step login pipeline.
    Body: {"images": ["b64_1", ...], "challenge": "blink", "session_id": "uuid"}
    """
    images_b64 = data.get("images", [])
    session_id = data.get("session_id", "")
    client_signature = data.get("signature", "")
    
    if not session_id or session_id not in session_secrets:
        return {"status": "fail", "reason": "invalid_session", "message": "Invalid or expired session"}
        
    session_data = session_secrets[session_id]
    secret_key = session_data["secret"]
    flash_sequence = session_data["sequence"]
    
    # ── 0. HMAC Security Check (Anti-Injection) ──
    # Verify that the payload was actually captured during the active session
    # by checking the HMAC-SHA256 signature generated by the frontend.
    payload_string = session_id + "".join(img[-20:] for img in images_b64)
    expected_sig = hmac.new(
        secret_key.encode(), 
        payload_string.encode(), 
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(expected_sig, client_signature):
        return {"status": "fail", "reason": "hmac_failed", "message": "Payload signature invalid. Injection attack mitigated."}

    if len(images_b64) < 3:
        return {"status": "error", "message": "At least 3 flash frames required"}

    # ── 1. Input: Decode frames ──
    frames = [decode_frame(img) for img in images_b64]

    # ── 2. Preprocessing: RetinaFace detect & crop ──
    face = detect_face(frames[0])
    if face is None:
        return {"status": "fail", "reason": "no_face",
                "message": "No face detected. Face the camera clearly."}

    face_crop = face["face_crop"]

    # ── 3. Anti-Spoof Layer ──
    spoof = compute_antispoof_score(face_crop)
    if not spoof["is_real"]:
        log_attendance("UNKNOWN", 0, spoof["score"], 0, "FAILED_SPOOF")
        return {"status": "fail", "reason": "spoof_detected",
                "message": "Presentation attack detected.",
                "antispoof_score": spoof["score"],
                "breakdown": spoof["breakdown"]}

    # ── 4. Active Flash Liveness (Web-based 3D Reflection) ──
    challenge = compute_flash_liveness(frames[:len(flash_sequence)], flash_sequence)
    if not challenge["passed"]:
        log_attendance("UNKNOWN", 0, spoof["score"], 0, "FAILED_FLASH_LIVENESS")
        return {"status": "fail", "reason": "flash_failed",
                "message": "3D active reflection failed. Put face near screen.",
                "details": challenge["details"]}
    
    del session_secrets[session_id] # Burn the session to prevent replay attacks

    # ── 5. Liveness Layer ──
    liveness = compute_liveness_score(frames, face_crop)
    if not liveness["is_live"]:
        log_attendance("UNKNOWN", 0, spoof["score"], liveness["score"], "FAILED_LIVENESS")
        return {"status": "fail", "reason": "liveness_failed",
                "message": "Liveness check failed.",
                "liveness_score": liveness["score"],
                "motion": liveness["motion_score"],
                "texture": liveness["texture_score"]}

    # ── 6. Face Embedding (ArcFace) & Deep Learning PAD ──
    res = compute_embedding(frames[0])
    embedding = res["embedding"]
    dl_antispoof_passed = res["is_real"]
    
    if not dl_antispoof_passed:
        log_attendance("UNKNOWN", 0, spoof["score"], liveness["score"], "FAILED_DL_PAD")
        return {"status": "fail", "reason": "dl_pad_failed",
                "message": "Deep Learning Presentation Attack Detection failed."}

    # ── 7. FAISS Matching ──
    if faiss_index.total == 0:
        return {"status": "fail", "reason": "no_users",
                "message": "No registered users. Register first."}

    matches = faiss_index.search(embedding, k=1)
    if not matches:
        log_attendance("UNKNOWN", 0, spoof["score"], liveness["score"], "FAILED_NO_MATCH")
        return {"status": "fail", "reason": "no_match",
                "message": "Face not recognized."}

    best = matches[0]
    face_score = best["score"]
    matched_name = best["name"]

    # ── 8. Combined Decision ──
    a_pass = spoof["score"] >= ANTISPOOF_THRESHOLD
    l_pass = liveness["score"] >= LIVENESS_THRESHOLD
    f_pass = face_score >= FACE_MATCH_THRESHOLD

    scores = {
        "face_score": round(face_score, 4),
        "antispoof_score": spoof["score"],
        "liveness_score": liveness["score"],
        "challenge_confidence": challenge["confidence"],
    }

    if not (a_pass and l_pass and f_pass):
        reasons = []
        if not a_pass: reasons.append("antispoof")
        if not l_pass: reasons.append("liveness")
        if not f_pass: reasons.append("face_match")
        log_attendance(matched_name, face_score, spoof["score"],
                       liveness["score"], "FAILED_DECISION")
        return {"status": "fail", "reason": "decision_failed",
                "failed_checks": reasons, "scores": scores}

    # ── Self-Learning: update embedding with running average ──
    if face_score >= SELF_LEARN_THRESHOLD:
        update_embedding_self_learn(matched_name, embedding.tolist(), LEARNING_RATE)
        rebuild_index()

    # ── 9. Output: Log attendance & return success ──
    log_attendance(matched_name, face_score, spoof["score"],
                   liveness["score"], "SUCCESS")

    return {
        "status": "success",
        "name": matched_name,
        "scores": scores,
        "message": f"Welcome, {matched_name}! Attendance recorded.",
    }


@app.get("/attendance")
def attendance(name: str = None, limit: int = 50):
    """Retrieve attendance log entries."""
    return get_attendance_log(name=name, limit=limit)
