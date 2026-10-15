"""backend.manage_keys -- DEPRECATED.

The per-officer API-key system this script managed (table `api_keys`, `backend.main.hash_api_key`) was removed:
officer identity is no longer a provisioned credential. Login is now a single predefined passkey per role
(ADMIN_PASSKEY / ANALYST_PASSKEY, set in .env) plus a self-reported name typed at sign-in -- see backend/main.py's
module docstring and the POST /api/login handler. There is nothing left for this script to create, list, or
revoke, and the old `api_keys` table is no longer created by SCHEMA.

Kept only so `docker compose exec backend python manage_keys.py ...` fails with a clear message instead of an
ImportError, in case an old runbook or shell history still calls it.
"""
from __future__ import annotations

import sys


def main() -> None:
    sys.stderr.write(
        "manage_keys.py is deprecated and does nothing: per-officer API keys were replaced by the shared\n"
        "admin/analyst passkey login (POST /api/login). Set ADMIN_PASSKEY and ANALYST_PASSKEY in .env instead.\n"
        "See backend/main.py's module docstring for details.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
