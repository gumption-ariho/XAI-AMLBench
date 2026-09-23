"""GNN detection API (container: gnn-detection-api).

Endpoints
  GET  /health            liveness + model status
  GET  /model/info        training metrics, decision threshold, targets
  GET  /accounts/sample   a few suspicious / benign account ids to try
  POST /predict           risk scores for one or more accounts (2-hop subgraph inference)
  POST /explain           GNNExplainer subgraph + edge importances (input for xai-narrative-api)

Before starting, train a model:
  python -m aml_synth.graph_generator --out data --to csv
  python -m gnn_aml_core.train --data data --out models
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from pydantic import BaseModel, Field
from torch_geometric.utils import k_hop_subgraph

from gnn_aml_core.models import build_model
from xai_explainer.extractor import extract_explanation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gnn_aml_core.api")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
LATENCY_TARGET_MS = float(os.getenv("GNN_LATENCY_TARGET_MS", 100))
AUC_TARGET = float(os.getenv("GNN_AUC_TARGET", 0.87))
CACHE_TTL = int(os.getenv("PRED_CACHE_TTL", 300))
REPORTING_THRESHOLD = float(os.getenv("REPORTING_THRESHOLD", 10_000))
MAX_EXPLAIN_EDGES = 8000        # counts both the forward and the reversed copy of every transaction
MAX_SCORE_EDGES = int(os.getenv("MAX_SCORE_EDGES", 30_000))   # higher budget than explaining: one forward pass
                                                                # is far cheaper per edge than GNNExplainer's
                                                                # repeated optimisation passes

# ---- custom Prometheus metrics (scraped by prometheus-metrics -> Grafana) ----
FLAGGED = Counter("aml_flagged_accounts_total", "Accounts scored above the alert threshold")
CACHE_HITS = Counter("aml_prediction_cache_hits_total", "Predictions served from Redis")
INFER_LAT = Histogram("aml_inference_latency_seconds", "End-to-end /predict latency",
                      buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
AUC_GAUGE = Gauge("aml_model_auc_roc", "Test AUC-ROC of the loaded model")


class State:
    model = None
    ckpt: dict | None = None
    g: dict | None = None
    id2idx: dict[str, int] = {}
    hops = 2
    calibrator = None             # set at load time: Calibrator from the checkpoint, or an identity fallback
    redis = None
    immu = None
    lock = threading.Lock()      # torch modules + GNNExplainer mask hooks are not thread-safe


S = State()


# ------------------------------------------------------------------ startup
def _load_model() -> None:
    mp, gp = MODEL_DIR / "gnn_model.pt", MODEL_DIR / "graph.pt"
    if not (mp.exists() and gp.exists()):
        log.warning("no model found in %s -- run gnn_aml_core.train first", MODEL_DIR)
        return
    # These files are produced by our own train.py, so full unpickling is acceptable here.
    ckpt = torch.load(mp, map_location="cpu", weights_only=False)
    g = torch.load(gp, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["in_dim"], **ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    from gnn_aml_core.calibration import Calibrator
    # Older checkpoints (trained before calibration was added) have no "calibration" key: fall back to an
    # identity mapping so loading an old model.pt still works, just without the calibration benefit.
    S.calibrator = Calibrator.from_dict(ckpt["calibration"]) if "calibration" in ckpt else Calibrator.identity()
    S.model, S.ckpt, S.g = model, ckpt, g
    S.id2idx = {a: i for i, a in enumerate(g["account_ids"])}
    S.hops = int(ckpt["hparams"].get("num_layers", 2))
    AUC_GAUGE.set(ckpt["metrics"]["test_auc_roc"])
    log.info("loaded %s: %d nodes, %d edges, threshold %.3f",
             ckpt["model_name"], len(S.id2idx), g["edge_index"].size(1), ckpt["threshold"])


def _connect_redis() -> None:
    url = os.getenv("REDIS_URL")
    if not url:
        return
    try:
        import redis
        r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.25)
        r.ping()
        S.redis = r
        log.info("redis feature/prediction cache connected")
    except Exception as exc:  # noqa: BLE001
        log.warning("redis unavailable (%s) -- continuing without cache", exc.__class__.__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    _connect_redis()
    yield


app = FastAPI(title="XAI-AMLBench GNN Detection API", version="0.1.0", lifespan=lifespan)
_inst = Instrumentator()
_inst.add(metrics.latency(buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)))
_inst.add(metrics.requests())
_inst.instrument(app).expose(app, include_in_schema=False)


# ------------------------------------------------------------------ helpers
def _require_model() -> None:
    if S.model is None:
        raise HTTPException(503, "Model not loaded. Run: python -m gnn_aml_core.train")


def _idx(account_id: str) -> int:
    if account_id not in S.id2idx:
        raise HTTPException(404, f"unknown account_id '{account_id}'")
    return S.id2idx[account_id]


def _cache_get(account_id: str) -> float | None:
    if S.redis is None:
        return None
    try:
        v = S.redis.get(f"pred:{account_id}")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


def _cache_set(account_id: str, p: float) -> None:
    if S.redis is None:
        return
    try:
        S.redis.setex(f"pred:{account_id}", CACHE_TTL, f"{p:.6f}")
    except Exception:  # noqa: BLE001
        pass


def audit(kind: str, payload: dict) -> None:
    """Append an event to immudb (tamper-evident ledger). Never raises."""
    host = os.getenv("IMMUDB_HOST")
    if not host:
        return
    try:
        if S.immu is None:
            from immudb import ImmudbClient
            client = ImmudbClient(f"{host}:{os.getenv('IMMUDB_PORT', '3322')}")
            client.login(os.getenv("IMMUDB_USER", "immudb"), os.getenv("IMMUDB_ADMIN_PASSWORD", "immudb"))
            S.immu = client
        record = {"kind": kind, "ts": time.time(), **payload}
        key = f"{kind}:{payload.get('account_id', '-')}:{time.time_ns()}".encode()
        S.immu.verifiedSet(key, json.dumps(record, sort_keys=True).encode())
    except Exception as exc:  # noqa: BLE001
        log.warning("immudb audit failed (%s)", exc.__class__.__name__)
        S.immu = None


def _score(indices: list[int]) -> list[float]:
    """Run the model on the union of the k-hop in-neighbourhoods of the requested nodes.

    Falls back to fewer hops if the combined subgraph would be unreasonably large: without this, a single
    /predict request for an account near a high-degree hub (e.g. a smurfing collection account with thousands
    of transactions) pulls in the hub's entire neighbourhood and can turn one request that should take
    milliseconds into one that takes seconds. Mirrors the same protection /explain already has (MAX_EXPLAIN_EDGES),
    just with a higher budget, since a single forward pass is far cheaper per edge than GNNExplainer's repeated
    optimisation passes. Reducing hops does reduce the model's receptive field for that call -- a deliberate,
    rare trade-off for the handful of very large hub accounts, not something that affects ordinary accounts."""
    g = S.g
    idx_t = torch.tensor(indices)
    with S.lock, torch.no_grad():
        hops = S.hops
        subset, ei, mapping, emask = k_hop_subgraph(idx_t, hops, g["edge_index"], relabel_nodes=True, num_nodes=g["x"].size(0))
        while ei.size(1) > MAX_SCORE_EDGES and hops > 1:
            hops -= 1
            subset, ei, mapping, emask = k_hop_subgraph(idx_t, hops, g["edge_index"], relabel_nodes=True, num_nodes=g["x"].size(0))
        out = S.model(g["x"][subset], ei, g["edge_attr"][emask], g["edge_type"][emask])
        return torch.sigmoid(out[mapping]).tolist()


# ------------------------------------------------------------------ schemas
class PredictRequest(BaseModel):
    account_ids: list[str] = Field(min_length=1, max_length=200)


class ExplainRequest(BaseModel):
    account_id: str
    top_k: int = Field(40, ge=5, le=200)
    epochs: int = Field(100, ge=20, le=300)


# ------------------------------------------------------------------ routes
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": S.model is not None, "redis": S.redis is not None,
            "model": S.ckpt["model_name"] if S.ckpt else None}


@app.get("/model/info")
def model_info():
    _require_model()
    return {"model": S.ckpt["model_name"], "hparams": S.ckpt["hparams"], "threshold": S.ckpt["threshold"],
            "metrics": S.ckpt["metrics"], "targets": {"auc_roc": AUC_TARGET, "latency_ms": LATENCY_TARGET_MS},
            "n_nodes": len(S.id2idx), "n_features": len(S.ckpt["feature_names"])}


@app.get("/accounts/sample")
def sample(n: int = Query(5, ge=1, le=50)):
    _require_model()
    y = S.g["y"]
    sus = (y == 1).nonzero().view(-1)[:n].tolist()
    ben = (y == 0).nonzero().view(-1)[:n].tolist()
    ids = S.g["account_ids"]
    return {"suspicious": [ids[i] for i in sus], "benign": [ids[i] for i in ben]}


@app.post("/predict")
def predict(req: PredictRequest):
    _require_model()
    t0 = time.perf_counter()
    scores: dict[str, tuple[float, bool]] = {}
    todo: list[str] = []
    for a in dict.fromkeys(req.account_ids):
        _idx(a)  # 404 if unknown
        cached = _cache_get(a)
        if cached is not None:
            scores[a] = (cached, True)
            CACHE_HITS.inc()
        else:
            todo.append(a)
    if todo:
        for a, p in zip(todo, _score([S.id2idx[a] for a in todo])):
            scores[a] = (p, False)
            _cache_set(a, p)

    thr = S.ckpt["threshold"]
    results = []
    for a, (p, cached) in scores.items():
        flagged = p >= thr
        # "score" (raw) drives the flagged decision and stays backward-compatible; "calibrated_score" is the
        # same account mapped through isotonic calibration -- meaningful as a probability, meant for display
        # (a risk gauge, a narrative's "risk score: X%"), and never used for the flagged decision itself, so
        # existing behaviour and caching (keyed on the raw score) are unaffected.
        results.append({"account_id": a, "score": round(p, 5), "calibrated_score": round(S.calibrator(p), 5),
                        "flagged": flagged, "cached": cached})
        if flagged:
            FLAGGED.inc()
            audit("inference_flagged", {"account_id": a, "score": round(p, 5),
                                        "model": S.ckpt["model_name"], "threshold": thr})
    dt = time.perf_counter() - t0
    INFER_LAT.observe(dt)
    return {"results": results, "threshold": thr, "calibrated_threshold": S.ckpt.get("calibrated_threshold", thr),
            "latency_ms": round(dt * 1000, 2), "meets_latency_target": dt * 1000 <= LATENCY_TARGET_MS}


@app.post("/explain")
def explain(req: ExplainRequest):
    _require_model()
    if S.ckpt["model_name"] != "gatv2":
        raise HTTPException(501, "GNNExplainer is wired for the GATv2 model. Train with --model gatv2.")
    _idx(req.account_id)  # 404 if unknown, before we take the lock

    with S.lock:
        try:
            result = extract_explanation(
                S.model, S.g, S.ckpt["feature_names"], req.account_id,
                hops=S.hops, reporting_threshold=REPORTING_THRESHOLD,
                top_k=req.top_k, epochs=req.epochs, max_explain_edges=MAX_EXPLAIN_EDGES,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    audit("explanation_generated", {"account_id": req.account_id, "score": result["risk_score"],
                                    "n_edges": len(result["edges"]), "method": "GNNExplainer"})
    # Replace the raw risk_score with its calibrated value for display (a risk gauge showing "99.9%" for
    # every flagged account is not informative); the paired threshold is calibrated the same way, so anything
    # comparing risk_score against threshold stays on a consistent scale. The raw value stays available as
    # raw_risk_score for anyone who needs the model's unmodified output (e.g. audit records already logged it).
    return {**result, "risk_score": round(S.calibrator(result["risk_score"]), 5), "raw_risk_score": result["risk_score"],
            "threshold": S.ckpt.get("calibrated_threshold", S.ckpt["threshold"]), "model": S.ckpt["model_name"], "method": "GNNExplainer"}
