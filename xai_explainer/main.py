"""XAI narrative API (container: xai-narrative-api).

POST /narrative  {account_id}  or  {account_id, explanation}
  1. gets a GNNExplainer subgraph from gnn-detection-api (unless one is supplied)
  2. reduces it to verified facts  (sar_generator.build_facts)
  3. asks the local quantized Llama-3-8B (vLLM, temperature 0) to write 4 sentences
  4. validates the text against the fact sheet; on any problem returns the deterministic template instead

Set LLM_BACKEND=template to run without a GPU (template narratives only).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from pydantic import BaseModel, Field

from xai_explainer import sar_generator as sar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("xai_explainer")

GNN_API_URL = os.getenv("GNN_API_URL", "http://gnn-detection-api:8000")
LLM_BACKEND = os.getenv("LLM_BACKEND", "vllm")            # vllm | template
LLM_MODEL = os.getenv("LLM_MODEL", "casperhansen/llama-3-8b-instruct-awq")
LLM_QUANT = os.getenv("LLM_QUANT", "awq") or None
LLM_MAX_LEN = int(os.getenv("LLM_MAX_MODEL_LEN", 4096))

NARRATIVES = Counter("sar_narratives_total", "SAR narratives produced", ["source", "validated"])


class State:
    llm = None
    lock = threading.Lock()      # vLLM's offline LLM object must not be called concurrently
    immu = None


S = State()


def _load_llm() -> None:
    if LLM_BACKEND != "vllm":
        log.info("LLM_BACKEND=%s -> template narratives only", LLM_BACKEND)
        return
    try:
        from vllm import LLM
        S.llm = LLM(model=LLM_MODEL, quantization=LLM_QUANT, max_model_len=LLM_MAX_LEN,
                    gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL", 0.85)), seed=0)
        log.info("vLLM loaded %s", LLM_MODEL)
    except Exception as exc:  # noqa: BLE001
        log.error("could not load vLLM (%s: %s) -- falling back to template narratives", exc.__class__.__name__, exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_llm()
    yield


app = FastAPI(title="XAI-AMLBench Narrative API", version="0.1.0", lifespan=lifespan)
_inst = Instrumentator()
_inst.add(metrics.latency(buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)))
_inst.add(metrics.requests())
_inst.instrument(app).expose(app, include_in_schema=False)


def audit(kind: str, payload: dict) -> None:
    host = os.getenv("IMMUDB_HOST")
    if not host:
        return
    try:
        if S.immu is None:
            from immudb import ImmudbClient
            c = ImmudbClient(f"{host}:{os.getenv('IMMUDB_PORT', '3322')}")
            c.login(os.getenv("IMMUDB_USER", "immudb"), os.getenv("IMMUDB_ADMIN_PASSWORD", "immudb"))
            S.immu = c
        key = f"{kind}:{payload.get('account_id', '-')}:{time.time_ns()}".encode()
        S.immu.verifiedSet(key, json.dumps({"kind": kind, "ts": time.time(), **payload}, sort_keys=True).encode())
    except Exception as exc:  # noqa: BLE001
        log.warning("immudb audit failed (%s)", exc.__class__.__name__)
        S.immu = None


def _llm_narrative(fact_sheet: str) -> str:
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=260, seed=0)
    with S.lock:
        out = S.llm.chat(sar.build_messages(fact_sheet), params, use_tqdm=False)
    return out[0].outputs[0].text.strip()


class NarrativeRequest(BaseModel):
    account_id: str
    explanation: dict | None = Field(default=None, description="optional /explain payload; fetched if omitted")
    use_llm: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "llm_loaded": S.llm is not None, "backend": LLM_BACKEND, "model": LLM_MODEL}


@app.post("/narrative")
def narrative(req: NarrativeRequest):
    explanation = req.explanation
    if explanation is None:
        try:
            r = httpx.post(f"{GNN_API_URL}/explain", json={"account_id": req.account_id}, timeout=120)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"gnn-detection-api unreachable: {exc.__class__.__name__}") from exc
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text)
        explanation = r.json()

    facts = sar.build_facts(explanation)
    sheet = sar.render_fact_sheet(facts)

    text, source, valid, problems = None, "template", True, []
    if req.use_llm and S.llm is not None:
        try:
            candidate = _llm_narrative(sheet)
            valid, problems = sar.validate_narrative(candidate, sheet)
            if valid:
                text, source = candidate, "llm"
            else:
                log.warning("LLM narrative rejected for %s: %s", req.account_id, problems)
        except Exception as exc:  # noqa: BLE001
            log.error("LLM generation failed: %s", exc)
            problems = [f"llm error: {exc.__class__.__name__}"]
    if text is None:
        text = sar.template_narrative(facts)
        valid, _ = sar.validate_narrative(text, sheet)

    digest = sar.narrative_hash(text)
    NARRATIVES.labels(source=source, validated=str(valid).lower()).inc()
    audit("sar_narrative", {"account_id": req.account_id, "source": source, "model": LLM_MODEL if source == "llm" else "template",
                            "narrative_sha256": digest, "risk_score": explanation["risk_score"]})
    return {
        "account_id": req.account_id, "narrative": text, "source": source, "validated": valid,
        "rejected_llm_problems": problems if source == "template" else [],
        "typology": facts["typology"], "risk_score": explanation["risk_score"],
        "fact_sheet": sheet, "narrative_sha256": digest,
    }
