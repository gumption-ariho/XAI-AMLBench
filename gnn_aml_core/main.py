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

from gnn_aml_core.features import merge_reverse_copies
from gnn_aml_core.models import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gnn_aml_core.api")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
LATENCY_TARGET_MS = float(os.getenv("GNN_LATENCY_TARGET_MS", 100))
AUC_TARGET = float(os.getenv("GNN_AUC_TARGET", 0.87))
CACHE_TTL = int(os.getenv("PRED_CACHE_TTL", 300))
REPORTING_THRESHOLD = float(os.getenv("REPORTING_THRESHOLD", 10_000))
MAX_EXPLAIN_EDGES = 8000        # counts both the forward and the reversed copy of every transaction

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
    """Run the model on the union of the k-hop in-neighbourhoods of the requested nodes."""
    g = S.g
    with S.lock, torch.no_grad():
        subset, ei, mapping, emask = k_hop_subgraph(
            torch.tensor(indices), S.hops, g["edge_index"], relabel_nodes=True, num_nodes=g["x"].size(0))
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
        results.append({"account_id": a, "score": round(p, 5), "flagged": flagged, "cached": cached})
        if flagged:
            FLAGGED.inc()
            audit("inference_flagged", {"account_id": a, "score": round(p, 5),
                                        "model": S.ckpt["model_name"], "threshold": thr})
    dt = time.perf_counter() - t0
    INFER_LAT.observe(dt)
    return {"results": results, "threshold": thr, "latency_ms": round(dt * 1000, 2),
            "meets_latency_target": dt * 1000 <= LATENCY_TARGET_MS}


@app.post("/explain")
def explain(req: ExplainRequest):
    _require_model()
    if S.ckpt["model_name"] != "gatv2":
        raise HTTPException(501, "GNNExplainer is wired for the GATv2 model. Train with --model gatv2.")
    from torch_geometric.explain import Explainer, GNNExplainer

    g, idx = S.g, _idx(req.account_id)
    n = g["x"].size(0)

    with S.lock:
        # The stored graph is bidirectional (every transaction also has a reversed copy), so a k-hop query
        # returns the complete neighbourhood; very large hubs fall back to 1 hop.
        subset, ei, mapping, emask = k_hop_subgraph(idx, S.hops, g["edge_index"], relabel_nodes=True, num_nodes=n)
        if ei.size(1) > MAX_EXPLAIN_EDGES:
            subset, ei, mapping, emask = k_hop_subgraph(idx, 1, g["edge_index"], relabel_nodes=True, num_nodes=n)
        ea = g["edge_attr"][emask]
        et = g["edge_type"][emask]
        x_sub = g["x"][subset]
        center = int(mapping[0])

        with torch.no_grad():
            score = float(torch.sigmoid(S.model(x_sub, ei, ea, et)[center]))
        explainer = Explainer(
            model=S.model,
            algorithm=GNNExplainer(epochs=req.epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
        )
        expl = explainer(x_sub, ei, index=center, edge_attr=ea, edge_type=et)
        edge_imp = expl.edge_mask.detach().cpu()
        feat_imp = expl.node_mask[center].abs().detach().cpu()

    n_tx = int(g["n_tx"])
    global_edge_ids = emask.nonzero().view(-1).numpy()
    folded = merge_reverse_copies(global_edge_ids, ei[0].numpy(), ei[1].numpy(), edge_imp.numpy(), n_tx, req.top_k)
    node_ids = {center}
    edges = []
    for tx, s_l, d_l, imp in folded:
        node_ids.update((s_l, d_l))
        edges.append({
            "tx_id": g["tx_ids"][tx],
            "src": g["account_ids"][int(subset[s_l])],
            "dst": g["account_ids"][int(subset[d_l])],
            "amount": round(float(g["tx_amount"][tx]), 2),
            "timestamp": int(g["tx_timestamp"][tx]),
            "cross_border": int(g["tx_cross_border"][tx]),
            "importance": round(imp, 5),
        })
    nodes = []
    for l in sorted(node_ids):
        gi = int(subset[l])
        nodes.append({"account_id": g["account_ids"][gi], "account_type": g["account_type"][gi],
                      "country": g["country"][gi]})

    total = float(feat_imp.sum()) or 1.0
    top = torch.argsort(feat_imp, descending=True)[:6].tolist()
    top_features = [{"feature": S.ckpt["feature_names"][i], "weight": round(float(feat_imp[i]) / total, 4)}
                    for i in top if feat_imp[i] > 0]

    audit("explanation_generated", {"account_id": req.account_id, "score": round(score, 5),
                                    "n_edges": len(edges), "method": "GNNExplainer"})
    return {
        "account_id": req.account_id, "risk_score": round(score, 5), "threshold": S.ckpt["threshold"],
        "reporting_threshold": REPORTING_THRESHOLD, "model": S.ckpt["model_name"],
        "method": "GNNExplainer", "nodes": nodes, "edges": edges, "top_features": top_features,
    }
