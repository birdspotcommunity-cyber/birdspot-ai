import os
import base64
import json
import requests

from fastapi import UploadFile, Request

from app.prompts import SYSTEM_PROMPT
from app.cache import sha256_bytes, cache_get, cache_set
from app.spectrogram import audio_to_spectrogram_image
from app.species import match_species
from app.media_utils import resize_image, trim_audio
from app.usage_db import log_usage
from app.quotas import enforce_user_quota

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")


async def _call_openai_with_image(image_bytes: bytes, text: str = "") -> dict:
    """
    Calls OpenAI Chat Completions with a single image + optional prompt text.
    Returns a parsed JSON dict from the assistant output.
    If OpenAI returns non-JSON, it safely returns an "unknown" JSON object.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    if not text:
        text = "Identify the bird species in this image and return JSON."

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}"
                        },
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data = r.json()
    content = data["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Safeguard: never crash the endpoint
        return {
            "predictions": [
                {
                    "species_name": "unknown",
                    "scientific_name": "unknown",
                    "confidence": 0.34,
                    "reason": "Model returned non-JSON output.",
                },
                {
                    "species_name": "unknown",
                    "scientific_name": "unknown",
                    "confidence": 0.33,
                    "reason": "Model returned non-JSON output.",
                },
                {
                    "species_name": "unknown",
                    "scientific_name": "unknown",
                    "confidence": 0.33,
                    "reason": "Model returned non-JSON output.",
                },
            ],
            "notes": content[:4000],
        }


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

    photo_prompt = """
You are an expert ornithologist. Identify the bird species visible in the photo.

Return EXACTLY valid JSON in this format:

{
  "predictions": [
    {
      "species_name": "COMMON NAME (non-empty)",
      "scientific_name": "Scientific name (non-empty if possible)",
      "confidence": 0.0-1.0,
      "reason": "1-2 short sentences describing visible features"
    },
    {
      "species_name": "COMMON NAME (non-empty)",
      "scientific_name": "Scientific name (non-empty if possible)",
      "confidence": 0.0-1.0,
      "reason": "..."
    },
    {
      "species_name": "COMMON NAME (non-empty)",
      "scientific_name": "Scientific name (non-empty if possible)",
      "confidence": 0.0-1.0,
      "reason": "..."
    }
  ],
  "notes": "optional additional notes"
}

Rules:
- Return exactly 3 predictions.
- species_name MUST NOT be empty.
- scientific_name should be provided when possible.
- confidence must be between 0 and 1.
- reason MUST NOT be empty.
- Do not include any extra keys outside of this JSON object.
"""

    raw = await _call_openai_with_image(resized_bytes, photo_prompt)
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

    # Quota enforcement
    enforce_user_quota(request)

    # Generate bird-tuned spectrogram (Fix #2)
    spectro_png = audio_to_spectrogram_image(trimmed_wav)

    sound_prompt = """
You are an expert ornithologist.
You are looking at a LOG-FREQUENCY spectrogram image of a bird vocalization, tuned for 800Hz–11kHz.

Bird calls appear as:
- whistles (smooth lines),
- trills (rapid repeated lines),
- chirps (short bursts),
- harmonics (stacked lines).

Identify the most likely bird species from the pattern.

Return EXACTLY valid JSON in this format:

{
  "predictions": [
    {
      "species_name": "COMMON NAME (non-empty)",
      "scientific_name": "Scientific name (non-empty if possible)",
      "confidence": 0.0-1.0,
      "reason": "Describe what in the spectrogram suggests this species"
    },
    {
      "species_name": "COMMON NAME (non-empty)",
      "scientific_name": "Scientific name (non-empty if possible)",
      "confidence": 0.0-1.0,
      "reason": "..."
    },
    {
      "species_name": "COMMON NAME (non-empty)",
      "scientific_name": "Scientific name (non-empty if possible)",
      "confidence": 0.0-1.0,
      "reason": "..."
    }
  ],
  "notes": "optional notes about the call type, frequency range, or uncertainty"
}

Rules:
- Return exactly 3 predictions.
- species_name MUST NOT be empty.
- reason MUST NOT be empty.
- confidence must be between 0 and 1.
- Do not include any extra keys outside of this JSON object.
"""

    raw = await _call_openai_with_image(spectro_png, sound_prompt)

    normalized = _normalize_predictions(raw)
    normalized["cached"] = False
    normalized["input_bytes"] = len(trimmed_wav)

    cache_set(key, normalized)
    log_usage(user_id, ip, "/api/identify/sound", key, False, OPENAI_MODEL, len(trimmed_wav))
    return normalized
