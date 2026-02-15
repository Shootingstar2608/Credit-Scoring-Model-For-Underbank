"""
Authentication module — Login / Register / Logout
Simple file-based auth with hashed passwords (SHA-256).
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
USERS_FILE = Path(__file__).parent / "users.json"

# Default users (always available even if users.json is deleted)
DEFAULT_USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "Quản trị viên",
        "role": "admin",
        "created_at": "2026-01-01T00:00:00",
    },
    "demo": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Người dùng Demo",
        "role": "user",
        "created_at": "2026-01-01T00:00:00",
    },
}


# ============================================================
# HELPERS
# ============================================================
def _hash_password(password: str) -> str:
    """Hash password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict:
    """Load users from JSON file, merge with defaults."""
    users = dict(DEFAULT_USERS)  # Start with defaults
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            users.update(saved)  # Saved users override defaults
        except (json.JSONDecodeError, IOError):
            pass
    return users


def _save_users(users: dict) -> None:
    """Save users to JSON file."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


# ============================================================
# PUBLIC API
# ============================================================
def authenticate(username: str, password: str) -> dict | None:
    """
    Check username + password. Returns user dict if valid, None otherwise.
    """
    users = _load_users()
    user = users.get(username)
    if user and user["password_hash"] == _hash_password(password):
        return {"username": username, "name": user["name"], "role": user["role"]}
    return None


def register(username: str, password: str, name: str, role: str = "user") -> str | None:
    """
    Register a new user. Returns error message if failed, None if success.
    """
    username = username.strip().lower()
    if not username or not password or not name:
        return "Vui lòng điền đầy đủ thông tin."
    if len(username) < 3:
        return "Tên đăng nhập phải có ít nhất 3 ký tự."
    if len(password) < 6:
        return "Mật khẩu phải có ít nhất 6 ký tự."
    if not username.isalnum():
        return "Tên đăng nhập chỉ được chứa chữ và số."

    users = _load_users()
    if username in users:
        return f"Tên đăng nhập '{username}' đã tồn tại."

    users[username] = {
        "password_hash": _hash_password(password),
        "name": name,
        "role": role,
        "created_at": datetime.now().isoformat(),
    }
    _save_users(users)
    return None


def get_all_users() -> list[dict]:
    """Get all users (for admin panel). Returns list of user info (no passwords)."""
    users = _load_users()
    return [
        {
            "username": uname,
            "name": info["name"],
            "role": info["role"],
            "created_at": info.get("created_at", "N/A"),
        }
        for uname, info in users.items()
    ]


def delete_user(username: str) -> bool:
    """Delete a user (admin only). Cannot delete 'admin'."""
    if username == "admin":
        return False
    users = _load_users()
    if username in users:
        del users[username]
        _save_users(users)
        return True
    return False
