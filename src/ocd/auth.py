"""Minimal multi-user auth: accounts, password hashing, and in-memory sessions.

Stdlib only (PBKDF2-HMAC-SHA256) — no extra dependency. Accounts live in
``data/users/accounts.yaml`` (git-ignored; it holds password *hashes*, never plaintext).
Each account owns a workspace at ``data/users/<user>/`` used as its pipeline OCD_HOME.

This is lightweight auth meant for a trusted, local/LAN multi-user setup — not a
hardened public service. Sessions are kept in memory (cleared on restart) and the
dev server runs over plain HTTP; put it behind HTTPS before exposing it.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timezone

import yaml

from . import paths

USERNAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
RESERVED_USERNAMES = {"accounts", "admin", "root"}
MIN_PASSWORD_LEN = 4
_ITERATIONS = 200_000

# token -> username (in-memory; lost on restart)
_SESSIONS: dict[str, str] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def valid_username(username: str) -> bool:
    return bool(USERNAME_RE.match(username)) and username not in RESERVED_USERNAMES


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             bytes.fromhex(salt), _ITERATIONS)
    return dk.hex()


# --------------------------------------------------------------------------- #
# Account store
# --------------------------------------------------------------------------- #
def load_accounts() -> dict[str, dict]:
    if not paths.ACCOUNTS_YAML.exists():
        return {}
    data = yaml.safe_load(paths.ACCOUNTS_YAML.read_text()) or {}
    return dict(data.get("users", {}))


def save_accounts(users: dict[str, dict]) -> None:
    paths.USERS_DIR.mkdir(parents=True, exist_ok=True)
    paths.ACCOUNTS_YAML.write_text(yaml.safe_dump({"users": users}, sort_keys=True))
    try:
        os.chmod(paths.ACCOUNTS_YAML, 0o600)  # hashes — keep readable only by owner
    except OSError:
        pass


def user_exists(username: str) -> bool:
    return username.strip().lower() in load_accounts()


def create_user(username: str, password: str) -> str:
    """Create an account + its workspace. Returns the normalized username. Raises ValueError."""
    username = username.strip().lower()
    if not valid_username(username):
        raise ValueError("Username must be 1–32 chars: a–z, 0–9, '_' or '-'.")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    users = load_accounts()
    if username in users:
        raise ValueError("That username is already taken.")
    salt = secrets.token_hex(16)
    users[username] = {"salt": salt, "hash": _hash_password(password, salt),
                       "created_at": _now_iso()}
    save_accounts(users)
    (paths.user_home(username) / "data" / "statements").mkdir(parents=True, exist_ok=True)
    return username


def verify_password(username: str, password: str) -> bool:
    rec = load_accounts().get(username.strip().lower())
    if not rec:
        return False
    return hmac.compare_digest(_hash_password(password, rec["salt"]), rec["hash"])


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def new_session(username: str) -> str:
    token = secrets.token_urlsafe(24)
    _SESSIONS[token] = username
    return token


def session_user(token: str | None) -> str | None:
    return _SESSIONS.get(token) if token else None


def end_session(token: str | None) -> None:
    if token:
        _SESSIONS.pop(token, None)
