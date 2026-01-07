import os
import base64
import json
import requests
from fastapi import UploadFile, Request

from app.prompts_validate import SYSTEM_PROMPT_VALIDATE, USER_PROMPT_VALIDATE_TEMPLATE
from app.cache import sha256_bytes, cache_get, cache_set
from app.spectrogram import audio_to_spectrogram_image
from app.media_utils import trim_audio
from app.species import load_species
from app.usage_db import log_usage

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _call_openai_validate(spectrogram_png: bytes, prompt_text: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    b64 = base64.b64encode(spectrogram_png).decode("utf-8")
    image_url = f"data:image/png;base64,{b64}"

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_VALIDATE},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
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
    return json.loads(content)


def _species_by_id(species_id: str):
    all_species = load_species()
    for s in all_species:
        if s.get("id") == species_id:
            return s
    return None


def _make_candidates_block(candidate_ids: list[str]) -> str:
    lines = []
    for cid in candidate_ids:
        s = _species_by_id(cid)
        if s:
            lines.append(f"- {s['id']} | {s['species_name']} | {s['scientific_name']}")
        else:
            lines.append(f"- {cid} | unknown |")
    return "\n".join(lines)


async def validate_sound_against_candidates(
    request: Request,
    audio: UploadFile,
    target_species_id: str,
    candidate_species_ids: list[str],
    location: str = "",
    season: str = "",
    habitat: str = "",
) -> dict:
    raw_bytes = await audio.read()
    trimmed_wav = trim_audio(raw_bytes)

    key = "validate_" + sha256_bytes(
        trimmed_wav
        + target_species_id.encode("utf-8")
        + ",".join(candidate_species_ids).encode("utf-8")
    )
    cached = cache_get(key)

    user_id = request.headers.get(
        "x-user-id", f"ip:{request.client.host if request.client else 'unknown'}"
    )
    ip = request.headers.get(
        "x-forwarded-for",
        request.client.host if request.client else "unknown",
    ).split(",")[0].strip()

    if cached:
        cached["cached"] = True
        log_usage(user_id, ip, "/api/validate/sound", key, True, OPENAI_MODEL, len(trimmed_wav))
        return cached

    target = _species_by_id(target_species_id) or {
        "id": target_species_id,
        "species_name": "unknown",
        "scientific_name": "",
    }
    candidates_block = _make_candidates_block(candidate_species_ids)

    prompt = USER_PROMPT_VALIDATE_TEMPLATE.format(
        target_species_id=target.get("id", ""),
        target_species_name=target.get("species_name", ""),
        target_scientific_name=target.get("scientific_name", ""),
        candidates_block=candidates_block,
        location=location or "unknown",
        season=season or "unknown",
        habitat=habitat or "unknown",
    )

    spectrogram_png = audio_to_spectrogram_image(trimmed_wav)
    raw = _call_openai_validate(spectrogram_png, prompt)

    best_id = raw.get("best_match_species_id")
    alt_id = raw.get("best_alternative_species_id")

    best = _species_by_id(best_id) if best_id else None
    alt = _species_by_id(alt_id) if alt_id else None
    target_s = (
        _species_by_id(raw.get("target_species_id"))
        if raw.get("target_species_id")
        else target
    )

    out = {
        "target_species": target_s,
        "best_match": best
        if best
        else {"id": best_id, "species_name": "unknown", "scientific_name": ""},
        "best_alternative": alt
        if alt
        else {"id": alt_id, "species_name": "unknown", "scientific_name": ""},
        "match": raw.get("match", "uncertain"),
        "match_confidence": float(raw.get("match_confidence") or 0.0),
        "best_alternative_confidence": float(raw.get("best_alternative_confidence") or 0.0),
        "explanation": raw.get("explanation", ""),
        "cached": False,
        "input_bytes": len(trimmed_wav),
    }

    cache_set(key, out)
    log_usage(user_id, ip, "/api/validate/sound", key, False, OPENAI_MODEL, len(trimmed_wav))
    return out
