"""Tests for backend.main's authentication (hash_api_key, get_current_officer). A real security fix: the
backend previously had no authentication at all, and the officer field on a decision was free-text and
self-reported. Only the database call is mocked (no live Postgres in a unit test); everything else -- FastAPI's
own HTTPException, the real hash function, the real header-parsing logic -- runs for real.
"""
import os

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")  # required at import time; never
                                                                                # actually connected to in these tests

from fastapi import HTTPException

import backend.main as m


class TestHashApiKey:
    def test_same_input_gives_same_hash(self):
        assert m.hash_api_key("officer_abc123") == m.hash_api_key("officer_abc123")

    def test_different_input_gives_different_hash(self):
        assert m.hash_api_key("officer_abc123") != m.hash_api_key("officer_different")

    def test_output_is_a_sha256_hex_digest(self):
        h = m.hash_api_key("anything")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestGetCurrentOfficer:
    def test_missing_authorization_header_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            m.get_current_officer(authorization=None)
        assert exc.value.status_code == 401
        assert "missing or malformed" in exc.value.detail

    def test_wrong_auth_scheme_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            m.get_current_officer(authorization="Basic dGVzdA==")
        assert exc.value.status_code == 401

    def test_empty_key_after_bearer_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            m.get_current_officer(authorization="Bearer ")
        assert exc.value.status_code == 401
        assert "empty" in exc.value.detail

    def test_unknown_key_is_rejected(self, monkeypatch):
        class FakeConn:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a, **kw): return self
            def fetchone(self): return None   # no matching row: unknown or revoked key

        monkeypatch.setattr(m, "db", lambda: FakeConn())
        with pytest.raises(HTTPException) as exc:
            m.get_current_officer(authorization="Bearer sometotallyfakekey")
        assert exc.value.status_code == 401
        assert "invalid or revoked" in exc.value.detail

    def test_valid_key_returns_the_correct_officer_name(self, monkeypatch):
        class FakeConn:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, *a, **kw): return self
            def fetchone(self): return {"officer_name": "Jane Smith"}

        monkeypatch.setattr(m, "db", lambda: FakeConn())
        assert m.get_current_officer(authorization="Bearer officer_realkey123") == "Jane Smith"

    def test_query_excludes_revoked_keys_and_hashes_the_input_key(self, monkeypatch):
        captured = {}

        class FakeConn:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, q, args=None):
                captured["q"], captured["args"] = q, args
                return self
            def fetchone(self): return {"officer_name": "X"}

        monkeypatch.setattr(m, "db", lambda: FakeConn())
        m.get_current_officer(authorization="Bearer key123")
        assert "revoked_at IS NULL" in captured["q"]
        assert captured["args"] == (m.hash_api_key("key123"),)   # the RAW key must never be compared directly;
                                                                 # only its hash should reach the database
