"""
Lightweight account system for the "My Store" self-serve flow.

MVP-level security note: passwords are hashed with PBKDF2-HMAC-SHA256 and a
per-account random salt (not plaintext, not reversible) -- reasonable for a
small pilot, but this is NOT a substitute for a real auth provider
(e.g. Auth0, Firebase Auth, or a proper backend with HTTPS + rate limiting)
before selling this to real, paying customers at any scale. See the
"Turning This Into a Real Product" section of the README for the full
production checklist.
"""
import hashlib
import hmac
import json
import os
import re
import secrets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMERS_DIR = os.path.join(BASE_DIR, "data", "customers")
ACCOUNTS_FILE = os.path.join(CUSTOMERS_DIR, "accounts.json")

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{3,32}$")


def _load_accounts():
    if not os.path.exists(ACCOUNTS_FILE):
        return {}
    with open(ACCOUNTS_FILE) as f:
        return json.load(f)


def _save_accounts(accounts):
    os.makedirs(CUSTOMERS_DIR, exist_ok=True)
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(accounts, f, indent=2)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def register(username: str, password: str, store_name: str = "") -> tuple:
    """Returns (success: bool, message: str)."""
    if not USERNAME_RE.match(username):
        return False, "Username must be 3-32 characters: letters, numbers, - or _ only."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    accounts = _load_accounts()
    if username in accounts:
        return False, "That username is already taken."

    salt = secrets.token_hex(16)
    accounts[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "store_name": store_name or username,
    }
    _save_accounts(accounts)
    os.makedirs(account_dir(username), exist_ok=True)
    return True, "Account created."


def verify(username: str, password: str) -> bool:
    accounts = _load_accounts()
    if username not in accounts:
        return False
    acc = accounts[username]
    expected = acc["password_hash"]
    actual = _hash_password(password, acc["salt"])
    return hmac.compare_digest(expected, actual)


def get_store_name(username: str) -> str:
    accounts = _load_accounts()
    return accounts.get(username, {}).get("store_name", username)


def account_dir(username: str) -> str:
    """Each account's model/data lives in its own folder -- one store's
    data is never visible to or loadable by another account."""
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", username)
    d = os.path.join(CUSTOMERS_DIR, safe)
    os.makedirs(d, exist_ok=True)
    return d
