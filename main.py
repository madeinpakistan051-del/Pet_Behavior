# main.py
# Smart Paws — AI Behaviour Checker API
# Deploy on Railway.app

import os
print("🚀🚀🚀 SERVER IS RUNNING THE NEW GROQ CODE! 🚀🚀🚀")
import uuid
import shutil
import time
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq

from inference import PetBehaviorEngine

# ── Model paths ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))

XGB_PATH    = os.path.join(MODEL_DIR, "xgboost_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "flow_scaler.pkl")
CNN_PATH    = os.path.join(MODEL_DIR, "cnn_scripted.pt")

# ── Upload temp directory ────────────────────────────────────────────
UPLOAD_DIR = "/tmp/smart_paws_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Max video size: 50 MB ────────────────────────────────────────────
MAX_VIDEO_BYTES = 50 * 1024 * 1024

# ── Global engine ────────────────────────────────────────────────────
engine: PetBehaviorEngine = None

# ── Groq client ──────────────────────────────────────────────────────
api_key     = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key) if api_key else None
if not groq_client:
    print("WARNING: GROQ_API_KEY not found. Will fall back to raw model output.")


# ════════════════════════════════════════════════════════════════════
# STARTUP — verify models exist before loading
# ════════════════════════════════════════════════════════════════════

def verify_models():
    """Check all model files are present. Raise a clear error if not."""
    missing = []
    for path in [XGB_PATH, SCALER_PATH, CNN_PATH]:
        if not os.path.exists(path):
            missing.append(path)
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ Found {os.path.basename(path)} ({size_mb:.1f} MB)")

    if missing:
        for p in missing:
            print(f"  ✗ MISSING: {p}")
        raise FileNotFoundError(
            f"Model files not found: {missing}\n"
            "Make sure the 'models/' folder is committed to your repository."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    # 1. Verify all model files exist
    print("Verifying model files …")
    try:
        verify_models()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("API will start but /analyze will return errors.")
        yield
        return

    # 2. Load models into memory
    print("Loading models into memory …")
    try:
        engine = PetBehaviorEngine(
            xgb_path    = XGB_PATH,
            scaler_path = SCALER_PATH,
            cnn_path    = CNN_PATH,
        )
        print("✅ All models loaded. API is ready.")
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print("API will start but /analyze will return errors.")

    yield

    # Cleanup on shutdown
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    print("API shut down cleanly.")


# ════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Smart Paws — AI Behaviour Checker",
    description = "Analyzes pet behavior from text descriptions and video clips.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["GET", "POST"],
    allow_headers = ["*"],
)


# ════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "service": "Smart Paws AI Behaviour Checker",
        "version": "1.0.0",
        "status":  "running",
        "endpoints": {
            "analyze": "POST /behavior/analyze",
            "health":  "GET  /health",
            "docs":    "GET  /docs",
        },
    }


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  engine is not None,
        "model_version": "1.0.0",
        "timestamp":     int(time.time()),
    }


@app.post("/behavior/analyze")
async def analyze_behavior(
    description: Optional[str]        = Form(default=""),
    animal:      Optional[str]        = Form(default="unknown"),
    breed:       Optional[str]        = Form(default="unknown"),
    video:       Optional[UploadFile] = File(default=None),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please retry in a few seconds.")

    has_text  = bool(description and description.strip())
    has_video = video is not None and video.filename

    if not has_text and not has_video:
        return JSONResponse(
            status_code=400,
            content={"detail": "Please provide a description or upload a video."}
        )

    tmp_video_path = None
    start_time     = time.time()

    try:
        # ── Save uploaded video to temp ──────────────────────────────
        if has_video:
            content = await video.read()
            if len(content) > MAX_VIDEO_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Video too large. Please upload a clip under 50 MB."}
                )

            ext = os.path.splitext(video.filename)[-1].lower() or ".mp4"
            if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                return JSONResponse(
                    status_code=415,
                    content={"detail": f"Unsupported format '{ext}'. Use MP4, MOV, or AVI."}
                )

            tmp_video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
            with open(tmp_video_path, "wb") as f:
                f.write(content)

        # ── 1. Run local inference ───────────────────────────────────
        raw_result = engine.predict(
            text       = description or "",
            video_path = tmp_video_path,
            breed      = breed or "unknown",
            animal     = animal or "unknown",
        )

        # ── 2. Format with Groq ──────────────────────────────────────
        final_ui_result = None

        if groq_client:
            try:
                system_prompt = (
                    "You are an expert veterinary behaviorist. "
                    "Your ONLY job is to output a strictly formatted JSON object. "
                    "CRITICAL RULE: You MUST map the situation to one of these exact 3 diagnoses: "
                    "1. 'Separation Anxiety'\n2. 'Aggression / Hostility'\n3. 'Depression / Stress'.\n\n"
                    "REQUIRED JSON SCHEMA:\n"
                    "{\n"
                    '  "diagnosis": "String (Must be exactly one of the 3 conditions above)",\n'
                    '  "confidence": "String (Convert raw confidence to a percentage, e.g., 97%)",\n'
                    '  "indicators": [\n'
                    '    {"icon": "warning", "text": "Short symptom text", "color": "red"}\n'
                    "  ],\n"
                    '  "actions": [\n'
                    '    {"title": "Action title", "desc": "Detailed explanation."}\n'
                    "  ]\n"
                    "}\n"
                    "DO NOT use old keys like 'detected_behavior'. YOU MUST ONLY use the schema above."
                )

                raw_behavior = raw_result.get("detected_behavior", "Unknown")
                raw_conf     = raw_result.get("confidence", 0.0)

                user_prompt = (
                    f"User Description: {description}\n"
                    f"AI Detection: {raw_behavior}\n"
                    f"AI Confidence: {raw_conf}\n\n"
                    "Translate this data into the required JSON schema."
                )

                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    model           = "llama-3.3-70b-versatile",
                    response_format = {"type": "json_object"},
                )

                raw_content     = chat_completion.choices[0].message.content
                clean_content   = raw_content.replace("```json", "").replace("```", "").strip()
                final_ui_result = json.loads(clean_content)

            except Exception as e:
                print(f"Groq formatting failed, using fallback: {e}")

        # ── 3. Fallback if Groq fails ────────────────────────────────
        if not final_ui_result:
            final_ui_result = {
                "diagnosis":  str(raw_result.get("detected_behavior", "Unknown Behavior")).title(),
                "confidence": "Analysis Complete",
                "indicators": [{"icon": "warning", "text": "See recommendations below", "color": "orange"}],
                "actions":    [
                    {"title": "Suggestion", "desc": s}
                    for s in raw_result.get("suggestions", ["Consult a vet for further advice."])
                ],
            }

        final_ui_result["processing_time_ms"] = round((time.time() - start_time) * 1000)
        return JSONResponse(content=final_ui_result)

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Analysis failed: {str(e)}"}
        )

    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
