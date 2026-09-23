#!/usr/bin/env python3
"""
run_local.py  -  run the whole application from your Python venv, with NO Docker image builds and NO downloads.

The infrastructure (Postgres, Redis, immudb, ...) stays in the Docker containers you already have. The four application
services run straight from ~/Desktop/XAI-AMLBench/.venv and frontend/node_modules:

    GNN model service   :8001   (loads ./models/gnn_model.pt)
    narrative service   :8002   (CPU, validated template narratives)
    backend API         :8000
    website             :3000   -> open http://localhost:3000

    python3 run_local.py                # start everything, Ctrl+C stops everything
    python3 run_local.py --dry-run      # show what would run, start nothing
    python3 run_local.py --no-frontend  # only the three APIs
    python3 run_local.py --stop         # stop leftovers from an earlier run (e.g. the terminal was closed)

Needs (once):   docker compose -f docker-compose.yml -f docker-compose.local.yml up -d database redis-feature-cache immudb-audit-ledger
Logs:           ./logs/gnn.log, xai.log, backend.log, frontend.log
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
PIDFILE = LOGS / "run_local.pids"
INFRA = {"Postgres": 15432, "Redis": 16379, "immudb (audit ledger)": 13322}


def read_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def build_services(env: dict, py: str) -> list:
    """(name, port, cwd, command, extra_env, health_path, wait_seconds)"""
    pg = f"postgresql://{env.get('POSTGRES_USER', 'projectxy')}:{env.get('POSTGRES_PASSWORD', '')}@127.0.0.1:15432/{env.get('POSTGRES_DB', 'projectxy')}"
    audit = {"IMMUDB_HOST": "127.0.0.1", "IMMUDB_PORT": "13322", "IMMUDB_ADMIN_PASSWORD": env.get("IMMUDB_ADMIN_PASSWORD", "")}
    uv = lambda app, port: [py, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(port)]
    return [
        ("gnn", 8001, ROOT, uv("gnn_aml_core.main:app", 8001),
         {"MODEL_DIR": str(ROOT / "models"), "REDIS_URL": f"redis://:{env.get('REDIS_PASSWORD', '')}@127.0.0.1:16379/0", "PYTHONPATH": str(ROOT), **audit},
         "/health", 180),
        ("xai", 8002, ROOT, uv("xai_explainer.main:app", 8002),
         {"LLM_BACKEND": "template", "GNN_API_URL": "http://127.0.0.1:8001", "PYTHONPATH": str(ROOT), **audit}, "/health", 60),
        ("backend", 8000, ROOT / "backend", uv("main:app", 8000),
         {"DATABASE_URL": pg, "GNN_API_URL": "http://127.0.0.1:8001", "XAI_API_URL": "http://127.0.0.1:8002", **audit}, "/api/health", 60),
        ("frontend", 3000, ROOT / "frontend", ["npm", "run", "dev"],
         {"BACKEND_URL": "http://127.0.0.1:8000", "NEXT_PUBLIC_DEMO": "0"}, "/", 240),
    ]


def healthy(port: int, path: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False
    except Exception:  # noqa: BLE001
        return False


def stop_pids() -> int:
    n = 0
    if PIDFILE.exists():
        for line in PIDFILE.read_text().split():
            try:
                os.killpg(int(line), signal.SIGTERM)
                n += 1
            except (ProcessLookupError, PermissionError, ValueError):
                pass
        PIDFILE.unlink(missing_ok=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-frontend", action="store_true")
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()

    if a.stop:
        print(f"stopped {stop_pids()} process group(s)")
        return 0

    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable
    env = read_env(ROOT / ".env")
    services = [s for s in build_services(env, py) if not (a.no_frontend and s[0] == "frontend")]

    print("XAI-AMLBench: local run (no Docker builds)\n" + "=" * 60)
    problems = []
    if not (ROOT / "models" / "gnn_model.pt").exists():
        problems.append("no trained model in ./models  ->  .venv/bin/python -m gnn_aml_core.train --data data --out models")
    if not (ROOT / ".env").exists():
        problems.append("no .env file  ->  python3 setup_backend_v22.py")
    for name, port in INFRA.items():
        ok = port_open(port)
        print(f"  {'ok  ' if ok else 'MISSING'}  {name:<24} 127.0.0.1:{port}")
        if not ok and not a.dry_run:
            problems.append(f"{name} is not reachable on 127.0.0.1:{port}  ->  docker compose -f docker-compose.yml -f docker-compose.local.yml up -d database redis-feature-cache immudb-audit-ledger")
    for s in services:
        if port_open(s[1]) and not a.dry_run:
            problems.append(f"port {s[1]} ({s[0]}) is already in use  ->  python3 run_local.py --stop   (or stop the docker container that uses it)")
    if a.dry_run:
        for name, port, cwd, cmd, extra, path, _ in services:
            print(f"\n  [{name}] port {port}, health {path}\n    cwd {cwd}\n    $ {' '.join(cmd)}")
            for k, v in extra.items():
                shown = v if not any(x in k for x in ("PASSWORD", "URL")) else "***"
                print(f"      {k}={shown}")
        return 0
    if problems:
        print("\nFix these first:")
        for p in problems:
            print("  -", p)
        return 1

    LOGS.mkdir(exist_ok=True)
    procs = []

    def shutdown(*_):
        print("\nstopping ...")
        for name, p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        PIDFILE.unlink(missing_ok=True)
        time.sleep(1)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        for name, port, cwd, cmd, extra, path, wait in services:
            log = open(LOGS / f"{name}.log", "w")
            p = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **extra}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append((name, p))
            PIDFILE.write_text(" ".join(str(x.pid) for _, x in procs))
            print(f"\n  starting {name} on :{port} ...", end="", flush=True)
            t0 = time.time()
            while time.time() - t0 < wait:
                if p.poll() is not None:
                    print(f" FAILED (exited with code {p.returncode})\n\n--- last lines of logs/{name}.log ---")
                    print("".join((LOGS / f"{name}.log").read_text().splitlines(True)[-15:]))
                    shutdown()
                if healthy(port, path):
                    print(f" ready in {time.time() - t0:.0f}s")
                    break
                time.sleep(1.5)
                print(".", end="", flush=True)
            else:
                print(f" no answer after {wait}s; see logs/{name}.log")
                shutdown()
    except Exception as exc:  # noqa: BLE001
        print(f"\nerror: {exc}")
        shutdown()

    print("\n" + "=" * 60)
    print("  Everything is running.   Open  http://localhost:3000")
    print("  Check the links:  python3 check_stack_v1.py --base http://localhost:3000 --gnn http://127.0.0.1:8001 --xai http://127.0.0.1:8002")
    print("  Ctrl+C here stops all four services.")
    while True:
        for name, p in procs:
            if p.poll() is not None:
                print(f"\n{name} stopped unexpectedly (code {p.returncode}); see logs/{name}.log")
                shutdown()
        time.sleep(2)


if __name__ == "__main__":
    sys.exit(main())
