"""
test_security.py — Unit tests for user_auth/security.py
  • Password hashing & verification
  • JWT creation & decoding (access + refresh)
  • Token blacklisting & auto-cleanup
  • Rate limit (dev mode always True)

Run:
    pytest tests/test_security.py -v
"""

import sys
import os
import uuid
import pytest
import pytest_asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "user_auth"))

# Stub config before importing security
_fake_settings = MagicMock()
_fake_settings.SECRET_KEY = "test-secret-key-for-unit-tests-only"
_fake_settings.JWT_ALGORITHM = "HS256"
_fake_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30
_fake_settings.REFRESH_TOKEN_EXPIRE_DAYS = 7
sys.modules.setdefault("config", MagicMock())
sys.modules["config"].settings = _fake_settings

import security  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#  Password Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPasswordHashing:
    def test_hash_password_returns_string(self):
        h = security.hash_password("MyPass123!")
        assert isinstance(h, str)
        assert h != "MyPass123!"

    def test_verify_correct_password(self):
        h = security.hash_password("correct")
        assert security.verify_password("correct", h) is True

    def test_verify_wrong_password(self):
        h = security.hash_password("correct")
        assert security.verify_password("wrong", h) is False

    def test_different_hashes_for_same_password(self):
        h1 = security.hash_password("same")
        h2 = security.hash_password("same")
        assert h1 != h2  # bcrypt salts

    def test_empty_password_hashes(self):
        h = security.hash_password("")
        assert isinstance(h, str)


# ═══════════════════════════════════════════════════════════════════════════════
#  JWT Token Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestJWTTokens:
    def test_create_access_token(self):
        token = security.create_access_token("user-123", "a@b.com", ["user"])
        assert isinstance(token, str)
        assert len(token) > 20

    def test_create_refresh_token(self):
        token = security.create_refresh_token("user-123")
        assert isinstance(token, str)
        assert len(token) > 20

    @pytest.mark.asyncio
    async def test_decode_access_token(self):
        uid = str(uuid.uuid4())
        token = security.create_access_token(uid, "test@test.com", ["user"])
        payload = await security.decode_token(token, expected_type="access")
        assert payload["sub"] == uid
        assert payload["email"] == "test@test.com"
        assert payload["type"] == "access"
        assert "jti" in payload

    @pytest.mark.asyncio
    async def test_decode_refresh_token(self):
        uid = str(uuid.uuid4())
        token = security.create_refresh_token(uid)
        payload = await security.decode_token(token, expected_type="refresh")
        assert payload["sub"] == uid
        assert payload["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_decode_wrong_type_raises(self):
        token = security.create_access_token("u1", "e@e.com", ["user"])
        from jose import JWTError
        with pytest.raises(JWTError, match="Invalid token type"):
            await security.decode_token(token, expected_type="refresh")

    @pytest.mark.asyncio
    async def test_decode_invalid_token_raises(self):
        from jose import JWTError
        with pytest.raises(JWTError):
            await security.decode_token("not.a.valid.token")

    @pytest.mark.asyncio
    async def test_access_token_contains_roles(self):
        token = security.create_access_token("u1", "e@e.com", ["admin", "user"])
        payload = await security.decode_token(token)
        assert payload["roles"] == ["admin", "user"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Token Blacklist Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenBlacklist:
    def setup_method(self):
        security._blacklisted_tokens.clear()

    def test_not_blacklisted_by_default(self):
        assert security.is_token_blacklisted("random-jti") is False

    @pytest.mark.asyncio
    async def test_blacklist_token(self):
        jti = str(uuid.uuid4())
        exp = datetime.now(timezone.utc) + timedelta(hours=1)
        await security.blacklist_token(jti, exp)
        assert security.is_token_blacklisted(jti) is True

    @pytest.mark.asyncio
    async def test_blacklisted_expired_token_auto_cleans(self):
        jti = str(uuid.uuid4())
        exp = datetime.now(timezone.utc) - timedelta(hours=1)  # already expired
        await security.blacklist_token(jti, exp)
        assert security.is_token_blacklisted(jti) is False

    @pytest.mark.asyncio
    async def test_decode_blacklisted_token_raises(self):
        uid = str(uuid.uuid4())
        token = security.create_access_token(uid, "e@e.com", ["user"])
        from jose import jwt as jose_jwt
        payload = jose_jwt.decode(token, _fake_settings.SECRET_KEY,
                                  algorithms=[_fake_settings.JWT_ALGORITHM])
        exp_dt = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        await security.blacklist_token(payload["jti"], exp_dt)
        from jose import JWTError
        with pytest.raises(JWTError, match="revoked"):
            await security.decode_token(token)


# ═══════════════════════════════════════════════════════════════════════════════
#  Rate Limit Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_always_allows_in_dev(self):
        result = await security.check_rate_limit("login:1.2.3.4", 10, 60)
        assert result is True
