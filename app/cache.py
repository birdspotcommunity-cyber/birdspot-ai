import os
import hashlib
import json

CACHE_DIR = os.getenv("CACHE_DIR", "./cache")

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def cache_get(key: str):
    ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def cache_set(key: str, value: dict):
    ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
