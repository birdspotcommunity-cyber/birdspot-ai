import os
from datetime import datetime
from fastapi import HTTPException, Request

from app.usage_db import get_daily_count, increment_daily

DAILY_LIMIT_PER_USER = int(os.getenv("DAILY_LIMIT_PER_USER", "25"))
DAILY_LIMIT_PER_IP = int(os.getenv("DAILY_LIMIT_PER_IP", "100"))

def today_key() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def get_user_id(request: Request) -> str:
    user_id = request.headers.get("x-user-id")
    if user_id:
        return user_id.strip()
    return f"ip:{get_client_ip(request)}"

def enforce_user_quota(request: Request):
    user_id = get_user_id(request)
    day = today_key()

    count = get_daily_count(user_id, day)
    if count >= DAILY_LIMIT_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"Daily identification limit reached ({DAILY_LIMIT_PER_USER}/day)."
        )

    increment_daily(user_id, day)
