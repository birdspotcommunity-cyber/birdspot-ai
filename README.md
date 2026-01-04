# BirdSpot AI Identify API (FastAPI)

This backend provides:
- POST /api/identify/photo  (multipart field: image)
- POST /api/identify/sound  (multipart field: audio)
It returns top-3 bird species predictions with confidence, using OpenAI Vision.

## Local run
1) Install ffmpeg
2) Create venv and install deps:
   pip install -r requirements.txt
3) Copy .env.example -> .env and set OPENAI_API_KEY
4) Run:
   uvicorn app.main:app --reload --port 8000

## Deploy
Use Docker (Render/Railway/Fly.io). Set env vars in your platform.
