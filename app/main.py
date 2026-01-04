from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from app.identify import identify_from_photo, identify_from_audio
from app.quotas import enforce_user_quota
from app.usage_db import init_db, fetch_recent_logs

load_dotenv()
init_db()

REQUIRE_FRONTEND_API_KEY = os.getenv("REQUIRE_FRONTEND_API_KEY", "false").lower() == "true"
FRONTEND_API_KEY = os.getenv("FRONTEND_API_KEY", "")

app = FastAPI(title="BirdSpot AI Identify API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_frontend_key(request: Request):
    if not REQUIRE_FRONTEND_API_KEY:
        return
    key = request.headers.get("x-frontend-api-key")
    if not key or key != FRONTEND_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid frontend API key.")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/identify/photo")
async def identify_photo(request: Request, image: UploadFile = File(...)):
    check_frontend_key(request)
    enforce_user_quota(request)

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    return await identify_from_photo(request, image)

@app.post("/api/identify/sound")
async def identify_sound(request: Request, audio: UploadFile = File(...)):
    check_frontend_key(request)
    enforce_user_quota(request)

    if audio.content_type and not (audio.content_type.startswith("audio/") or audio.content_type == "application/octet-stream"):
        raise HTTPException(status_code=400, detail="File must be audio.")

    return await identify_from_audio(request, audio)

@app.get("/admin/usage/recent")
def admin_recent(limit: int = 50):
    rows = fetch_recent_logs(limit=limit)
    return {
        "logs": [
            {
                "user_id": r[0],
                "ip": r[1],
                "endpoint": r[2],
                "file_hash": r[3],
                "cached": bool(r[4]),
                "created_at": r[5],
                "model": r[6],
                "input_bytes": r[7],
            }
            for r in rows
        ]
    }
