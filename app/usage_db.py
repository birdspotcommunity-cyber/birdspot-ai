import os
import sqlite3
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "./data/usage.sqlite")

def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS usage_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        ip TEXT,
        endpoint TEXT,
        file_hash TEXT,
        cached INTEGER,
        created_at TEXT,
        model TEXT,
        input_bytes INTEGER
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_usage (
        user_id TEXT,
        day TEXT,
        count INTEGER,
        PRIMARY KEY(user_id, day)
    )
    """)

    conn.commit()
    conn.close()

def log_usage(user_id: str, ip: str, endpoint: str, file_hash: str, cached: bool, model: str, input_bytes: int):
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO usage_logs (user_id, ip, endpoint, file_hash, cached, created_at, model, input_bytes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        ip,
        endpoint,
        file_hash,
        1 if cached else 0,
        datetime.utcnow().isoformat(),
        model,
        input_bytes
    ))

    conn.commit()
    conn.close()

def get_daily_count(user_id: str, day: str) -> int:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT count FROM daily_usage WHERE user_id=? AND day=?", (user_id, day))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0

def increment_daily(user_id: str, day: str):
    conn = _connect()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO daily_usage (user_id, day, count)
    VALUES (?, ?, 1)
    ON CONFLICT(user_id, day) DO UPDATE SET count = count + 1
    """, (user_id, day))

    conn.commit()
    conn.close()

def reset_daily(user_id: str, day: str):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM daily_usage WHERE user_id=? AND day=?", (user_id, day))
    conn.commit()
    conn.close()

def fetch_recent_logs(limit: int = 50):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
    SELECT user_id, ip, endpoint, file_hash, cached, created_at, model, input_bytes
    FROM usage_logs
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows
