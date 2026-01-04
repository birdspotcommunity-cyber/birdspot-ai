import os
import base64
import json
import requests
from fastapi import UploadFile, Request

from app.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from app.cache import sha256_bytes, cache_get, cache_set
from app.spectrogram import audio_to_spectrogram_image
from app.species import match_species
from app.media_utils import resize_image, trim_audio
from app.usage_db import log_usage

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def _call_openai_with_image(image_bytes: bytes) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{b64}"

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT_TEMPLATE},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ],
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)

def _normalize_predictions(result: dict) -> dict:
    preds = result.get("predictions", [])
    normalized = []

    for p in preds[:3]:
        species_name = (p.get("species_name") or "").strip()
        scientific_name = (p.get("scientific_name") or "").strip()
        confidence = float(p.get("confidence") or 0.0)
        reason = (p.get("reason") or "").strip()

        match = match_species(species_name, scientific_name)

        normalized.append({
            "species_id": match.get("id") if match else None,
            "species_name": match.get("species_name") if match else species_name,
            "scientific_name": match.get("scientific_name") if match else scientific_name,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": reason,
            "matched_to_db": bool(match)
        })

    while len(normalized) < 3:
        normalized.append({
            "species_id": None,
            "species_name": "unknown",
            "scientific_name": "",
            "confidence": 0.1,
            "reason": "Not enough information to identify.",
            "matched_to_db": False
        })

    return {
        "predictions": normalized,
        "notes": result.get("notes", "")
    }

async def identify_from_photo(request: Request, image: UploadFile) -> dict:
    raw_bytes = await image.read()
    resized_bytes = resize_image(raw_bytes)

    key = f"photo_{sha256_bytes(resized_bytes)}"
    cached = cache_get(key)

    user_id = request.headers.get("x-user-id", f"ip:{request.client.host if request.client else 'unknown'}")
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown").split(",")[0].strip()

    if cached:
        cached["cached"] = True
        log_usage(user_id, ip, "/api/identify/photo", key, True, OPENAI_MODEL, len(resized_bytes))
        return cached

    raw = _call_openai_with_image(resized_bytes)
    normalized = _normalize_predictions(raw)
    normalized["cached"] = False
    normalized["input_bytes"] = len(resized_bytes)

    cache_set(key, normalized)
    log_usage(user_id, ip, "/api/identify/photo", key, False, OPENAI_MODEL, len(resized_bytes))
    return normalized

async def identify_from_audio(request: Request, audio: UploadFile) -> dict:
    raw_bytes = await audio.read()
    trimmed_wav = trim_audio(raw_bytes)

    key = f"audio_{sha256_bytes(trimmed_wav)}"
    cached = cache_get(key)

    user_id = request.headers.get("x-user-id", f"ip:{request.client.host if request.client else 'unknown'}")
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown").split(",")[0].strip()

    if cached:
        cached["cached"] = True
        log_usage(user_id, ip, "/api/identify/sound", key, True, OPENAI_MODEL, len(trimmed_wav))
        return cached

    spectrogram_png = audio_to_spectrogram_image(trimmed_wav)
    raw = _call_openai_with_image(spectrogram_png)

    normalized = _normalize_predictions(raw)
    normalized["cached"] = False
    normalized["input_bytes"] = len(trimmed_wav)

    cache_set(key, normalized)
    log_usage(user_id, ip, "/api/identify/sound", key, False, OPENAI_MODEL, len(trimmed_wav))
    return normalized
