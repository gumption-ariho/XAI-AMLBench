"""Tests for backend.main's authentication: the stateless, HMAC-signed session-token system issued by
POST /api/login (one predefined passkey per role -- ADMIN_PASSKEY / ANALYST_PASSKEY -- plus the person's own
name). This replaced an earlier per-officer, database-backed API-key system (hash_api_key / the api_keys table /
manage_keys.py) -- those are gone, and so are the tests that exercised them. Only the database call is mocked
where a route happens to touch it (login's audit() call); everything else -- FastAPI's own HTTPException, the
real HMAC signing/verification, the real header-parsing logic -- runs for real.
"""
import os
import time

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")  # required at import time; never
                                                                                # actually connected to in these tests

import pytest
from fastapi import HTTPException

import backend.main as m


class TestIssueAndDecodeSessionToken:
    def test_round_trips_name_and_role(self):
        token = m.issue_session_token("Jane Smith", "analyst")
        data = m._decode_session_token(token)
        assert data["name"] == "Jane Smith"
        assert data["role"] == "analyst"

    def test_different_logins_get_different_tokens(self):
        # iat (issued-at) differs even for the same name/role, so tokens are never literally identical
        t1 = m.issue_session_token("Jane Smith", "analyst")
        time.sleep(0.01)
        t2 = m.issue_session_token("Jane Smith", "analyst")
        assert t1 != t2

    def test_tampered_payload_is_rejected(self):
        token = m.issue_session_token("Jane Smith", "analyst")
        encoded, sig = token.split(".", 1)
        # flip the role by re-encoding a modified payload, keeping the ORIGINAL (now-mismatched) signature
        import base64
        import json
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())
        payload["role"] = "admin"
        tampered_encoded = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
        with pytest.raises(HTTPException) as exc:
            m._decode_session_token(f"{tampered_encoded}.{sig}")
        assert exc.value.status_code == 401
        assert "signature" in exc.value.detail

    def test_malformed_token_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            m._decode_session_token("not-even-shaped-like-a-token")
        assert exc.value.status_code == 401

    def test_expired_token_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "SESSION_MAX_AGE_S", 1)
        token = m.issue_session_token("Jane Smith", "analyst")
        # backend.main does `import time`, so m.time IS this file's own `time` module object (the same entry
        # in sys.modules) -- NOT a separate copy. Patching m.time.time with a lambda that itself calls
        # time.time() therefore calls the just-patched function again, every time, forever: a real
        # RecursionError, caught on a live run against this exact project (not a hypothetical edge case).
        # Capturing the real function BEFORE patching, and calling THAT captured reference instead of going
        # back through the (now-patched) module attribute, is what breaks the self-reference.
        real_time = time.time
        monkeypatch.setattr(m.time, "time", lambda: real_time() + 2)
        with pytest.raises(HTTPException) as exc:
            m._decode_session_token(token)
        assert exc.value.status_code == 401
        assert "expired" in exc.value.detail

    def test_signature_uses_the_configured_secret(self, monkeypatch):
        monkeypatch.setattr(m, "AUTH_TOKEN_SECRET", "secret-a")
        token = m.issue_session_token("Jane Smith", "analyst")
        monkeypatch.setattr(m, "AUTH_TOKEN_SECRET", "secret-b")
        with pytest.raises(HTTPException) as exc:
            m._decode_session_token(token)
        assert exc.value.status_code == 401


class TestGetCurrentOfficer:
    def test_missing_authorization_header_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        with pytest.raises(HTTPException) as exc:
            m.get_current_officer(authorization=None)
        assert exc.value.status_code == 401
        assert "missing or malformed" in exc.value.detail

    def test_wrong_auth_scheme_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        with pytest.raises(HTTPException) as exc:
            m.get_current_officer(authorization="Basic dGVzdA==")
        assert exc.value.status_code == 401

    def test_empty_token_after_bearer_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        with pytest.raises(HTTPException) as exc:
            m.get_current_officer(authorization="Bearer ")
        assert exc.value.status_code == 401
        assert "empty" in exc.value.detail

    def test_valid_token_returns_the_name_it_was_issued_for(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        token = m.issue_session_token("Jane Smith", "analyst")
        assert m.get_current_officer(authorization=f"Bearer {token}") == "Jane Smith"

    def test_caller_cannot_change_name_by_sending_a_different_one_later(self, monkeypatch):
        # the name is bound into the signed token at login -- nothing in a later request's headers or body can
        # override it, which is the entire point of signing the name in rather than trusting a free-text field
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        token = m.issue_session_token("Jane Smith", "analyst")
        assert m.get_current_officer(authorization=f"Bearer {token}") == "Jane Smith"


class TestGetCurrentRole:
    def test_returns_the_role_the_token_was_issued_with(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        token = m.issue_session_token("Jane Smith", "admin")
        assert m.get_current_role(authorization=f"Bearer {token}") == "admin"

    def test_missing_header_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        with pytest.raises(HTTPException) as exc:
            m.get_current_role(authorization=None)
        assert exc.value.status_code == 401


class TestLoginRoute:
    def test_admin_passkey_issues_an_admin_token(self, monkeypatch):
        monkeypatch.setattr(m, "ADMIN_PASSKEY", "letmein-admin")
        monkeypatch.setattr(m, "ANALYST_PASSKEY", "letmein-analyst")
        monkeypatch.setattr(m, "audit", lambda *a, **kw: None)  # no live DB in this test
        res = m.login(m.LoginRequest(passkey="letmein-admin", name="Jane Smith"))
        assert res["role"] == "admin"
        assert res["name"] == "Jane Smith"
        assert m._decode_session_token(res["token"])["role"] == "admin"

    def test_analyst_passkey_issues_an_analyst_token(self, monkeypatch):
        monkeypatch.setattr(m, "ADMIN_PASSKEY", "letmein-admin")
        monkeypatch.setattr(m, "ANALYST_PASSKEY", "letmein-analyst")
        monkeypatch.setattr(m, "audit", lambda *a, **kw: None)
        res = m.login(m.LoginRequest(passkey="letmein-analyst", name="Jane Smith"))
        assert res["role"] == "analyst"

    def test_wrong_passkey_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "ADMIN_PASSKEY", "letmein-admin")
        monkeypatch.setattr(m, "ANALYST_PASSKEY", "letmein-analyst")
        with pytest.raises(HTTPException) as exc:
            m.login(m.LoginRequest(passkey="guessing", name="Jane Smith"))
        assert exc.value.status_code == 401

    def test_blank_name_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "ADMIN_PASSKEY", "letmein-admin")
        with pytest.raises(HTTPException) as exc:
            m.login(m.LoginRequest(passkey="letmein-admin", name="   "))
        assert exc.value.status_code == 400

    def test_unset_passkey_never_matches_anything(self, monkeypatch):
        # os.getenv returns None when a passkey isn't configured at all -- req.passkey must never be able to
        # equal None (e.g. by some client sending passkey: null and pydantic coercing it to the string "None")
        monkeypatch.setattr(m, "ADMIN_PASSKEY", None)
        monkeypatch.setattr(m, "ANALYST_PASSKEY", None)
        with pytest.raises(HTTPException) as exc:
            m.login(m.LoginRequest(passkey="None", name="Jane Smith"))
        assert exc.value.status_code == 401


class TestWhoami:
    """whoami() is deliberately trivial -- it just echoes the officer name and bank get_current_officer /
    get_current_bank already resolved and verified (see TestGetCurrentOfficer and TestGetCurrentBank for that
    logic's real tests). This confirms the route itself returns the correct shape, not a re-test of
    authentication logic that already has its own tests."""

    def test_returns_the_authenticated_officer_name_and_bank(self):
        assert m.whoami(officer="Jane Smith", bank=None) == {"officer": "Jane Smith", "bank": None}
        assert m.whoami(officer="Jane Smith", bank="Acme Bank") == {"officer": "Jane Smith", "bank": "Acme Bank"}


class TestGetCurrentBank:
    """The tenant boundary every bank-scoped route depends on alongside get_current_officer. None means "sees
    every bank's data" (admin/legacy-analyst/demo-bypass); a bank name means a BANK_PASSKEYS tenant session."""

    def test_admin_or_legacy_analyst_token_has_no_bank(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        token = m.issue_session_token("Jane Smith", "admin")
        assert m.get_current_bank(authorization=f"Bearer {token}") is None

    def test_bank_tenant_token_returns_its_bank(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        token = m.issue_session_token("Jane Smith", "analyst", "Acme Bank")
        assert m.get_current_bank(authorization=f"Bearer {token}") == "Acme Bank"

    def test_a_token_issued_before_this_field_existed_is_treated_as_no_bank(self, monkeypatch):
        # issue_session_token's `bank` param defaults to None, so this is really the same case as the admin
        # test above, but it's worth its own test: a session token minted by an older build of this service
        # (no "bank" key in its payload at all) must still decode cleanly and come back as "sees everything",
        # not crash whoami/every other route with a KeyError on first use after an upgrade.
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        token = m.issue_session_token("Jane Smith", "admin")
        assert m.get_current_bank(authorization=f"Bearer {token}") is None

    def test_demo_bypass_has_no_bank(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", True)
        assert m.get_current_bank(authorization=None) is None

    def test_missing_header_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        with pytest.raises(HTTPException) as exc:
            m.get_current_bank(authorization=None)
        assert exc.value.status_code == 401


class TestBankPasskeyLogin:
    """BANK_PASSKEYS logins: the multi-tenant SaaS door alongside the existing ADMIN_PASSKEY/ANALYST_PASSKEY
    ones. A bank passkey issues the same 'analyst' role the shared legacy passkey does, but binds a specific
    bank into the token -- see this module's own docstring on BANK_PASSKEYS for why role and tenant are
    separate concepts here."""

    def test_bank_passkey_issues_an_analyst_token_scoped_to_that_bank(self, monkeypatch):
        monkeypatch.setattr(m, "ADMIN_PASSKEY", "letmein-admin")
        monkeypatch.setattr(m, "ANALYST_PASSKEY", "letmein-analyst")
        monkeypatch.setattr(m, "PASSKEY_TO_BANK", {"acme-secret": "Acme Bank"})
        monkeypatch.setattr(m, "audit", lambda *a, **kw: None)
        res = m.login(m.LoginRequest(passkey="acme-secret", name="Jane Smith"))
        assert res["role"] == "analyst"
        assert res["bank"] == "Acme Bank"
        decoded = m._decode_session_token(res["token"])
        assert decoded["role"] == "analyst"
        assert decoded["bank"] == "Acme Bank"

    def test_different_banks_get_different_tenant_tokens(self, monkeypatch):
        monkeypatch.setattr(m, "ADMIN_PASSKEY", None)
        monkeypatch.setattr(m, "ANALYST_PASSKEY", None)
        monkeypatch.setattr(m, "PASSKEY_TO_BANK", {"acme-secret": "Acme Bank", "beta-secret": "Beta Bank"})
        monkeypatch.setattr(m, "audit", lambda *a, **kw: None)
        acme = m.login(m.LoginRequest(passkey="acme-secret", name="A"))
        beta = m.login(m.LoginRequest(passkey="beta-secret", name="B"))
        assert acme["bank"] == "Acme Bank"
        assert beta["bank"] == "Beta Bank"

    def test_admin_passkey_still_has_no_bank_even_when_bank_passkeys_are_configured(self, monkeypatch):
        monkeypatch.setattr(m, "ADMIN_PASSKEY", "letmein-admin")
        monkeypatch.setattr(m, "ANALYST_PASSKEY", None)
        monkeypatch.setattr(m, "PASSKEY_TO_BANK", {"acme-secret": "Acme Bank"})
        monkeypatch.setattr(m, "audit", lambda *a, **kw: None)
        res = m.login(m.LoginRequest(passkey="letmein-admin", name="Jane Smith"))
        assert res["role"] == "admin"
        assert res["bank"] is None

    def test_wrong_bank_passkey_is_rejected(self, monkeypatch):
        monkeypatch.setattr(m, "ADMIN_PASSKEY", None)
        monkeypatch.setattr(m, "ANALYST_PASSKEY", None)
        monkeypatch.setattr(m, "PASSKEY_TO_BANK", {"acme-secret": "Acme Bank"})
        with pytest.raises(HTTPException) as exc:
            m.login(m.LoginRequest(passkey="guessing", name="Jane Smith"))
        assert exc.value.status_code == 401


class TestCheckAccountAccess:
    """_check_account_access() is the cross-tenant guard every account-touching route (scan, explain) calls
    before doing anything else. lookup_bank() itself needs a real Postgres connection, so it's monkeypatched
    here the same way audit() is elsewhere in this file -- this tests the access-control logic, not the DB
    query it's layered on."""

    def test_admin_or_legacy_session_can_touch_any_account(self, monkeypatch):
        monkeypatch.setattr(m, "lookup_bank", lambda account_id: "Some Other Bank")
        m._check_account_access("ACC123", bank=None)   # must not raise

    def test_tenant_can_touch_its_own_banks_account(self, monkeypatch):
        monkeypatch.setattr(m, "lookup_bank", lambda account_id: "Acme Bank")
        m._check_account_access("ACC123", bank="Acme Bank")   # must not raise

    def test_tenant_can_touch_an_unattributed_demo_account(self, monkeypatch):
        # predates bank attribution (e.g. a SynthAML benchmark account) -- shared, not owned by anyone
        monkeypatch.setattr(m, "lookup_bank", lambda account_id: "unknown")
        m._check_account_access("ACC123", bank="Acme Bank")   # must not raise

    def test_tenant_cannot_touch_another_banks_account(self, monkeypatch):
        monkeypatch.setattr(m, "lookup_bank", lambda account_id: "Beta Bank")
        with pytest.raises(HTTPException) as exc:
            m._check_account_access("ACC123", bank="Acme Bank")
        assert exc.value.status_code == 404   # not 403 -- must not confirm the account exists under another bank


class TestDemoLoginBypass:
    """Off by default. A deliberate, clearly-labelled convenience for a reviewer demo -- see backend/main.py's
    own loud startup warning. Must never silently look like a real authenticated identity."""

    def test_off_by_default_still_requires_real_auth(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", False)
        with pytest.raises(HTTPException) as exc:
            m.get_current_officer(authorization=None)
        assert exc.value.status_code == 401

    def test_when_on_skips_auth_entirely_with_no_header_at_all(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", True)
        officer = m.get_current_officer(authorization=None)
        assert officer == "Reviewer (DEMO MODE)"

    def test_the_returned_identity_unmistakably_says_demo_mode(self, monkeypatch):
        # the whole point of this project's earlier security work was making sure the audit trail's identity
        # is always genuine -- this bypass must never produce a name that could pass as a real officer's
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", True)
        officer = m.get_current_officer(authorization=None)
        assert "DEMO" in officer.upper()

    def test_role_bypass_also_returns_a_fixed_safe_value(self, monkeypatch):
        monkeypatch.setattr(m, "DEMO_LOGIN_BYPASS", True)
        assert m.get_current_role(authorization=None) == "admin"


class TestIntakeClear:
    """POST /intake/clear -- Option 1's "clear temporary data" control. accounts_for_bank()/untag_accounts_bank()
    need a real Postgres connection, so they're monkeypatched the same way lookup_bank() is in
    TestCheckAccountAccess; _call() is monkeypatched so no real HTTP request to gnn-detection-api is made.
    This tests the tenant-forcing and keep-alerts/remove-accounts logic, not the DB query or HTTP call."""

    def _stub(self, monkeypatch, accounts, removed_calls, untagged_calls, audited):
        monkeypatch.setattr(m, "accounts_for_bank", lambda bank: accounts)
        monkeypatch.setattr(m, "untag_accounts_bank", lambda ids: untagged_calls.append(ids))
        monkeypatch.setattr(m, "_call", lambda method, url, **kw: (
            removed_calls.append((method, url, kw)),
            {"n_accounts": 3, "n_transactions": 7},
        )[1])
        monkeypatch.setattr(m, "audit", lambda kind, payload: audited.append((kind, payload)))

    def test_tenant_session_clears_its_own_bank_regardless_of_request_body(self, monkeypatch):
        removed_calls, untagged_calls, audited = [], [], []
        self._stub(monkeypatch, ["ACC1", "ACC2"], removed_calls, untagged_calls, audited)
        req = m.IntakeClearRequest(bank="Someone Else's Bank")   # must be ignored for a tenant session
        result = m.intake_clear(req, officer="Jane Smith", tenant_bank="Acme Bank")
        assert result["bank"] == "Acme Bank"
        assert result["removed_accounts"] == 2
        assert untagged_calls == [["ACC1", "ACC2"]]
        assert removed_calls[0][2]["json"] == {"account_ids": ["ACC1", "ACC2"]}
        assert audited[0][0] == "intake_clear"
        assert audited[0][1]["bank"] == "Acme Bank"

    def test_admin_session_must_name_a_bank(self, monkeypatch):
        removed_calls, untagged_calls, audited = [], [], []
        self._stub(monkeypatch, [], removed_calls, untagged_calls, audited)
        req = m.IntakeClearRequest(bank=None)
        with pytest.raises(HTTPException) as exc:
            m.intake_clear(req, officer="Jane Smith", tenant_bank=None)
        assert exc.value.status_code == 422

    def test_admin_session_clears_the_named_bank(self, monkeypatch):
        removed_calls, untagged_calls, audited = [], [], []
        self._stub(monkeypatch, ["ACC9"], removed_calls, untagged_calls, audited)
        req = m.IntakeClearRequest(bank="Beta Bank")
        result = m.intake_clear(req, officer="Admin User", tenant_bank=None)
        assert result["bank"] == "Beta Bank"
        assert result["removed_accounts"] == 1

    def test_nothing_to_clear_is_a_no_op_and_never_calls_remove_live(self, monkeypatch):
        removed_calls, untagged_calls, audited = [], [], []
        self._stub(monkeypatch, [], removed_calls, untagged_calls, audited)
        req = m.IntakeClearRequest(bank=None)
        result = m.intake_clear(req, officer="Jane Smith", tenant_bank="Acme Bank")
        assert result["status"] == "nothing to clear"
        assert result["removed_accounts"] == 0
        assert removed_calls == []        # never reached gnn-detection-api
        assert untagged_calls == []       # never touched account_banks
        assert audited == []              # never audited -- nothing happened

    def test_never_touches_alerts_or_anomaly_alerts_tables(self, monkeypatch):
        # The confirmed design: clearing removes accounts from the live graph only. The route's own body
        # never references the `alerts`/`anomaly_alerts` tables at all -- assert that directly against the
        # source rather than mocking a DB, so a future edit that starts touching them breaks this test.
        import inspect
        src = inspect.getsource(m.intake_clear)
        assert "DELETE FROM alerts" not in src
        assert "DELETE FROM anomaly_alerts" not in src
        assert "UPDATE alerts" not in src
        assert "UPDATE anomaly_alerts" not in src
