"""Admin credential validation, short-lived tokens, and login throttling."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Mapping

import bcrypt
from itsdangerous import BadSignature, URLSafeTimedSerializer

from backend.settings import (
    ADMIN_CREDENTIALS_JSON,
    ADMIN_TOKEN_SECRET,
    ADMIN_TOKEN_TTL_SECONDS,
)


MAX_LOGIN_FAILURES = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60
_BCRYPT_HASH = re.compile(r"^\$2[aby]\$(\d{2})\$[./A-Za-z0-9]{53}$")
_login_failures: dict[str, list[float]] = {}
_login_lock = threading.Lock()


class AuthConfigurationError(RuntimeError):
    """Admin authentication is missing or incorrectly configured."""


class InvalidAccessToken(ValueError):
    """The supplied bearer token is invalid or expired."""


def _configured_credentials() -> dict[str, bytes]:
    if not ADMIN_CREDENTIALS_JSON:
        raise AuthConfigurationError("Admin credentials are not configured.")
    try:
        raw_credentials = json.loads(ADMIN_CREDENTIALS_JSON)
    except json.JSONDecodeError as exc:
        raise AuthConfigurationError(
            "Admin credentials must be a JSON username-to-hash object."
        ) from exc
    if not isinstance(raw_credentials, Mapping) or not raw_credentials:
        raise AuthConfigurationError("At least one admin credential is required.")

    credentials: dict[str, bytes] = {}
    for username, encoded_hash in raw_credentials.items():
        if (
            not isinstance(username, str)
            or not username.strip()
            or username != username.strip()
            or len(username) > 128
            or not isinstance(encoded_hash, str)
        ):
            raise AuthConfigurationError("Admin credential entries are invalid.")
        match = _BCRYPT_HASH.fullmatch(encoded_hash)
        if not match or int(match.group(1)) < 12:
            raise AuthConfigurationError(
                "Admin passwords must use bcrypt hashes with a work factor of at least 12."
            )
        credentials[username] = encoded_hash.encode("ascii")
    return credentials


def admin_usernames() -> list[str]:
    return sorted(_configured_credentials())


def verify_admin_credentials(username: str, password: str) -> bool:
    credentials = _configured_credentials()
    encoded_hash = credentials.get(username)
    try:
        password_bytes = password.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if encoded_hash is None or len(password_bytes) > 72:
        return False
    return bcrypt.checkpw(password_bytes, encoded_hash)


def _serializer() -> URLSafeTimedSerializer:
    if len(ADMIN_TOKEN_SECRET.encode("utf-8")) < 32:
        raise AuthConfigurationError(
            "The admin token signing secret must contain at least 32 bytes."
        )
    return URLSafeTimedSerializer(
        ADMIN_TOKEN_SECRET,
        salt="drifting-oracle-admin-session",
    )


def create_access_token(username: str) -> str:
    return _serializer().dumps({"sub": username})


def verify_access_token(token: str) -> str:
    try:
        payload = _serializer().loads(
            token,
            max_age=ADMIN_TOKEN_TTL_SECONDS,
        )
    except BadSignature as exc:
        raise InvalidAccessToken("Invalid or expired admin session.") from exc
    username = payload.get("sub") if isinstance(payload, dict) else None
    if not isinstance(username, str) or username not in _configured_credentials():
        raise InvalidAccessToken("Invalid or expired admin session.")
    return username


def login_is_rate_limited(key: str) -> bool:
    now = time.monotonic()
    cutoff = now - LOGIN_LOCKOUT_SECONDS
    with _login_lock:
        failures = [
            timestamp
            for timestamp in _login_failures.get(key, [])
            if timestamp > cutoff
        ]
        if failures:
            _login_failures[key] = failures
        else:
            _login_failures.pop(key, None)
        return len(failures) >= MAX_LOGIN_FAILURES


def record_login_failure(key: str) -> None:
    now = time.monotonic()
    cutoff = now - LOGIN_LOCKOUT_SECONDS
    with _login_lock:
        failures = [
            timestamp
            for timestamp in _login_failures.get(key, [])
            if timestamp > cutoff
        ]
        failures.append(now)
        _login_failures[key] = failures
        if len(_login_failures) > 10_000:
            for old_key, timestamps in list(_login_failures.items()):
                if not any(timestamp > cutoff for timestamp in timestamps):
                    _login_failures.pop(old_key, None)


def clear_login_failures(key: str) -> None:
    with _login_lock:
        _login_failures.pop(key, None)


def reset_login_limits() -> None:
    """Clear in-memory throttling state, primarily for isolated tests."""
    with _login_lock:
        _login_failures.clear()
