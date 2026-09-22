#!/usr/bin/env python3
"""
check_brief_v1.py  -  the "XAI-AMLBench GitHub Project Brief" as an executable scorecard.

Every acceptance criterion of the brief is a PASS / FAIL check here. Run it, fix what fails, run it again.
Nothing is changed by this script (it only reads the project and writes to a temp folder).

    python3 check_brief_v1.py                 # all offline checks (about 20-60 seconds)
    python3 check_brief_v1.py --only B,C      # only some sections (A..G)
    python3 check_brief_v1.py --api http://localhost/gnn     # also test the RUNNING stack (latency, model loaded)
    python3 check_brief_v1.py --full          # also generate a 50,000-node graph (slow)
    python3 check_brief_v1.py --json report.json   # save the result, e.g. to track progress or attach to evidence
    python3 check_brief_v1.py --report-only   # never exit with an error code (default: exit 1 if anything FAILS)

Sections
  A  Repository, packaging and publishing files        E  Module 3: subgraph extraction, SAR narrative, audit log
  B  Module 1: synthetic generator + exporters         F  Tests and coverage (>= 80%)
  C  Is the benchmark hard enough? (baselines)         G  README and documentation
  D  Module 2: GNN metrics, baselines, latency

Run it from the project root (the folder that contains aml_synth/, gnn_aml_core/, xai_explainer/).
Needs numpy, pandas, networkx, scikit-learn (installed by install_all_requirements_v2.sh).
Checks that need PyTorch or a running API say SKIP instead of guessing.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# ---- thresholds copied from the brief ------------------------------------------------------------------------------
BRIEF = dict(auc=0.87, precision=0.89, recall=0.82, f1=0.85, fpr=0.02, latency_ms=100, coverage=80,
             median_amount=(500, 2000), node_tolerance=0.10, sar_tokens=150, sar_sentences=(3, 4),
             baseline_auc={"Logistic Regression": (0.65, 0.85), "XGBoost (gradient boosting)": (0.72, 0.92),
                           "Isolation Forest": (0.55, 0.75)},
             smurf_deposits=1000, big_graph_nodes=50_000)

RESULTS = []   # (section, status, name, detail)
SECTIONS = {"A": "Repository, packaging and publishing files", "B": "Module 1: synthetic generator + exporters",
            "C": "Is the benchmark hard enough? (baselines)", "D": "Module 2: GNN metrics, baselines, latency",
            "E": "Module 3: extractor, SAR narrative, audit log", "F": "Tests and coverage",
            "G": "README and documentation"}


def rec(sec, ok, name, detail=""):
    RESULTS.append((sec, "PASS" if ok else "FAIL", name, str(detail)))


def skip(sec, name, why):
    RESULTS.append((sec, "SKIP", name, why))


def guarded(sec, fn, *a):
    try:
        fn(*a)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        rec(sec, False, f"section {sec} crashed", f"{exc.__class__.__name__}: {exc} ({Path(tb.filename).name}:{tb.lineno})")


def any_glob(*patterns):
    return any(glob.glob(str(ROOT / p), recursive=True) for p in patterns)


def read(path):
    p = ROOT / path
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def est_tokens(text):
    """Conservative Llama-3-style estimate: words + extra for numbers and account ids."""
    return int(len(text.split()) * 1.25 + len(re.findall(r"\d+", text)) * 1.5 + len(re.findall(r"ACC\d+", text)) * 3)


# ================================================================================================= A
def section_a():
    S = "A"
    py = read("pyproject.toml")
    rec(S, bool(py), "pyproject.toml exists (PyPI packaging)", "" if py else "missing")
    if py:
        rec(S, bool(re.search(r'name\s*=\s*"xai-amlbench"', py)), 'package name is "xai-amlbench"')
        rec(S, bool(re.search(r'version\s*=\s*"0\.1\.0"', py)), 'version is "0.1.0"')
    lic = read("LICENSE")
    rec(S, "MIT License" in lic, "LICENSE is the MIT license", "missing" if not lic else "")
    rec(S, (ROOT / "CITATION.cff").exists(), "CITATION.cff (how to cite)")
    rec(S, (ROOT / ".zenodo.json").exists(), ".zenodo.json (Zenodo metadata: title, CC-BY-4.0 dataset licence, creators)")
    rec(S, any_glob(".github/workflows/*.yml", ".github/workflows/*.yaml"), "GitHub Actions workflow (test on push + coverage)")
    rec(S, (ROOT / "CONTRIBUTING.md").exists(), "CONTRIBUTING.md")
    rec(S, any_glob("notebooks/*.ipynb", "**/example*.ipynb"), "example notebook (Zenodo upload needs one)")
    rec(S, any_glob("**/huggingface*/README.md", "**/DATASET_CARD.md", "hub/**/dataset_card*.md"), "Hugging Face dataset card")
    rec(S, any_glob("**/track_metrics.py", "hub/metrics/*.py"), "weekly metrics tracker (stars, forks, PyPI, Zenodo, HF)")
    reqs = "".join(read(f) for f in ("requirements.txt", "aml_synth/requirements.txt", "gnn_aml_core/requirements.txt",
                                     "xai_explainer/requirements.txt", "pyproject.toml"))
    for pkg in ("pyarrow", "xgboost", "pytest", "pytest-cov", "networkx", "torch_geometric", "scikit-learn"):
        rec(S, pkg.lower().replace("_", "-") in reqs.lower().replace("_", "-"), f"dependency declared: {pkg}")


# ================================================================================================= B
def section_b(full):
    S = "B"
    from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig, START_TS, DAY, TYPOLOGIES
    import networkx as nx
    import pandas as pd

    n_req, days = 5000, 30
    g = AMLGraphGenerator(GeneratorConfig(n_accounts=n_req, n_background_tx=n_req * 8, days=days)).generate()
    A, T = g.accounts, g.transactions

    ratio = len(A) / n_req
    rec(S, abs(ratio - 1) <= BRIEF["node_tolerance"], "node count within +/-10% of the requested number", f"{len(A):,} vs {n_req:,} ({100 * (ratio - 1):+.1f}%)")
    G = g.to_networkx() if hasattr(g, "to_networkx") else nx.MultiDiGraph(list(zip(T.src, T.dst)))
    rec(S, G.is_directed() and G.is_multigraph(), "valid directed multigraph (networkx)", f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    for label, cols in (("node", ["account_type", "country", "risk_score"]), ):
        for c in cols:
            rec(S, c in A.columns, f"node attribute: {c}")
    rec(S, any(c in A.columns for c in ("creation_date", "opened_ts")), "node attribute: creation date")
    for c in ("amount", "timestamp"):
        rec(S, c in T.columns, f"edge attribute: {c}")
    rec(S, any(c in T.columns for c in ("transaction_type", "payment_format")), "edge attribute: transaction type")

    med = float(T.amount.median())
    lo, hi = BRIEF["median_amount"]
    rec(S, lo <= med <= hi, "median transaction amount is $500-$2,000", f"median is ${med:,.0f}")
    got = set(T[T.is_laundering == 1].typology.unique())
    rec(S, set(TYPOLOGIES) <= got, "all 5 laundering typologies present", f"found: {sorted(got)}")
    sm = T[T.typology == "smurfing"].groupby("pattern_id").size()
    rec(S, len(sm) > 0 and int(sm.max()) >= BRIEF["smurf_deposits"], "smurfing pattern reaches 1,000+ small deposits (brief: 1,000-2,000)",
        f"largest pattern has {int(sm.max()) if len(sm) else 0} transactions")

    bad = 0
    seeds = 12
    for s in range(seeds):
        t = AMLGraphGenerator(GeneratorConfig(n_accounts=1500, n_background_tx=6000, n_patterns_per_typology=40, days=days, seed=s)).generate().transactions
        bad += int((t.timestamp > START_TS + days * DAY).any() or (t.timestamp < START_TS).any())
    rec(S, bad == 0, f"all timestamps inside the date range, on {seeds} random seeds", f"{bad}/{seeds} seeds produced transactions outside the range")

    h1 = int(pd.util.hash_pandas_object(AMLGraphGenerator(GeneratorConfig(n_accounts=800, n_background_tx=3000, seed=7)).generate().transactions, index=False).sum())
    h2 = int(pd.util.hash_pandas_object(AMLGraphGenerator(GeneratorConfig(n_accounts=800, n_background_tx=3000, seed=7)).generate().transactions, index=False).sum())
    rec(S, h1 == h2, "same seed gives an identical graph (reproducible)")

    from aml_synth import exporters
    for fmt in ("csv", "parquet", "json", "cypher"):
        fn = getattr(exporters, f"export_{fmt}", None)
        if fn is None:
            rec(S, False, f"exporter: {fmt}", f"export_{fmt}() not implemented")
            continue
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(g, d)
                made = [p for p in Path(d).rglob("*") if p.is_file() and p.stat().st_size > 0]
                rec(S, bool(made), f"exporter: {fmt}", f"wrote {len(made)} file(s)" if made else "wrote nothing")
            except Exception as exc:  # noqa: BLE001
                rec(S, False, f"exporter: {fmt}", f"{exc.__class__.__name__}: {exc}")

    if full:
        t0 = time.time()
        big = AMLGraphGenerator(GeneratorConfig(n_accounts=BRIEF["big_graph_nodes"], n_background_tx=BRIEF["big_graph_nodes"] * 8)).generate()
        dt = time.time() - t0
        rec(S, dt < 180, f"a {BRIEF['big_graph_nodes']:,}-node graph generates in under 3 minutes", f"{len(big.accounts):,} nodes in {dt:.0f}s")
    else:
        skip(S, "50,000-node graph generates in reasonable time", "run with --full")


# ================================================================================================= C
def section_c():
    S = "C"
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig
    from gnn_aml_core.features import build_node_features, standardize

    g = AMLGraphGenerator(GeneratorConfig()).generate()
    X, _ = build_node_features(g.accounts, g.transactions)
    y = g.accounts.is_suspicious.values
    Xs, _, _ = standardize(X)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=1)
    aucs = {
        "Logistic Regression": roc_auc_score(y[te], LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]),
        "XGBoost (gradient boosting)": roc_auc_score(y[te], HistGradientBoostingClassifier(random_state=1).fit(Xs[tr], y[tr]).predict_proba(Xs[te])[:, 1]),
        "Isolation Forest": roc_auc_score(y[te], -IsolationForest(random_state=1, contamination=0.05).fit(Xs[tr]).score_samples(Xs[te])),
    }
    for name, (lo, hi) in BRIEF["baseline_auc"].items():
        rec(S, lo <= aucs[name] <= hi, f"{name} baseline AUC is in the brief's range {lo:.2f}-{hi:.2f}", f"measured {aucs[name]:.3f}")
    best = max(aucs.values())
    rec(S, best < BRIEF["auc"], f"headroom: best simple baseline stays below the GNN target ({BRIEF['auc']})",
        f"best baseline {best:.3f}; if this fails the benchmark cannot show that the GNN helps")
    rec(S, aucs["XGBoost (gradient boosting)"] > aucs["Logistic Regression"] - 0.02, "boosting is at least as good as logistic regression (sanity)")
    globals()["_BASELINES"] = aucs


# ================================================================================================= D
def section_d(api):
    S = "D"
    models = read("gnn_aml_core/models.py")
    rec(S, "class GATv2Detector" in models, "GATv2 model implemented")
    rec(S, "class RGCNDetector" in models, "RGCN model implemented")
    rec(S, (ROOT / "gnn_aml_core/baselines.py").exists(), "baselines.py (XGBoost, Logistic Regression, Isolation Forest)")
    tr = read("gnn_aml_core/train.py")
    rec(S, "pos_weight" in tr and re.search(r"19(\.0)?", tr) is not None, "class weight for positives is 19.0 (configurable)", "train.py computes it from the data instead" if "pos_weight" in tr else "")
    rec(S, "fpr" in tr.lower() or "false_positive" in tr.lower(), "training reports false-positive rate")
    rec(S, "f1" in tr.lower(), "training reports F1")

    metrics = None
    mj = ROOT / "models" / "metrics.json"
    if mj.exists():
        metrics = json.loads(mj.read_text())
    else:
        ck = ROOT / "models" / "gnn_model.pt"
        if ck.exists():
            try:
                import torch
                metrics = torch.load(ck, map_location="cpu", weights_only=False)["metrics"]
            except ImportError:
                skip(S, "GNN metrics from models/gnn_model.pt", "PyTorch is not installed in this Python (use .venv)")
    if metrics is None:
        skip(S, "GNN reaches AUC>=0.87, precision>=0.89, recall>=0.82, F1>=0.85, FPR<=0.02", "no trained model found: run python -m gnn_aml_core.train")
    else:
        auc = metrics.get("test_auc_roc", 0)
        pr, rc = metrics.get("test_precision", 0), metrics.get("test_recall", 0)
        f1 = metrics.get("test_f1", 2 * pr * rc / (pr + rc) if pr + rc else 0)
        rec(S, auc >= BRIEF["auc"], f"GNN AUC-ROC >= {BRIEF['auc']}", f"{auc:.3f}")
        rec(S, pr >= BRIEF["precision"], f"GNN precision >= {BRIEF['precision']}", f"{pr:.3f}")
        rec(S, rc >= BRIEF["recall"], f"GNN recall >= {BRIEF['recall']}", f"{rc:.3f}")
        rec(S, f1 >= BRIEF["f1"], f"GNN F1 >= {BRIEF['f1']}", f"{f1:.3f}")
        if "test_fpr" in metrics:
            rec(S, metrics["test_fpr"] <= BRIEF["fpr"], f"GNN false-positive rate <= {BRIEF['fpr']}", f"{metrics['test_fpr']:.3f}")
        else:
            rec(S, False, f"GNN false-positive rate <= {BRIEF['fpr']}", "not reported by training yet")
        base = globals().get("_BASELINES")
        if base:
            rec(S, auc > max(base.values()) + 0.02, "GNN beats the best simple baseline by a clear margin (the paper's claim)",
                f"GNN {auc:.3f} vs best baseline {max(base.values()):.3f}")

    if api:
        api = api.rstrip("/")
        try:
            def call(path, body=None):
                req = urllib.request.Request(api + path, data=json.dumps(body).encode() if body is not None else None,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return json.loads(r.read())
            h = call("/health")
            rec(S, bool(h.get("model_loaded")), "running API has the model loaded", str(h))
            ids = call("/accounts/sample?n=10")
            accounts = (ids["suspicious"] + ids["benign"])[:10]
            call("/predict", {"account_ids": accounts})                       # warm-up
            times = []
            for a in accounts * 2:
                t0 = time.perf_counter()
                r = call("/predict", {"account_ids": [a]})
                times.append(r.get("latency_ms", (time.perf_counter() - t0) * 1000))
            p95 = sorted(times)[int(len(times) * 0.95) - 1]
            rec(S, p95 < BRIEF["latency_ms"], f"inference latency < {BRIEF['latency_ms']} ms (p95 over {len(times)} calls)", f"median {statistics.median(times):.1f} ms, p95 {p95:.1f} ms")
        except Exception as exc:  # noqa: BLE001
            rec(S, False, "running API answers /health, /predict", f"{exc.__class__.__name__}: {exc}")
    else:
        skip(S, "inference latency < 100 ms on the running stack", "run with --api http://localhost/gnn")


# ================================================================================================= E
def section_e():
    S = "E"
    rec(S, (ROOT / "xai_explainer/extractor.py").exists(), "xai_explainer/extractor.py (GNNExplainer subgraph -> JSON)", "explanation code currently lives inside gnn_aml_core/main.py")
    rec(S, (ROOT / "xai_explainer/audit_log.py").exists(), "xai_explainer/audit_log.py (immutable audit trail)")

    import pandas as pd
    from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig, TYPOLOGIES
    from xai_explainer import sar_generator as sar

    g = AMLGraphGenerator(GeneratorConfig(n_accounts=1200, n_background_tx=5000, n_patterns_per_typology=4)).generate()
    A, T = g.accounts.set_index("account_id"), g.transactions
    ok_sent = ok_tok = ok_det = ok_val = ok_facts = True
    worst = 0
    detail = []
    for typ in TYPOLOGIES:
        pid = sorted(T[T.typology == typ].pattern_id.unique())[0]
        sub = T[T.pattern_id == pid]
        subject = pd.concat([sub.src, sub.dst]).value_counts().index[0]
        ids = list(dict.fromkeys([subject] + [x for r in sub.itertuples() for x in (r.src, r.dst)]))
        expl = {"account_id": subject, "risk_score": 0.92, "reporting_threshold": 10000,
                "nodes": [{"account_id": i, "account_type": A.loc[i, "account_type"], "country": A.loc[i, "country"]} for i in ids],
                "edges": [{"tx_id": r.tx_id, "src": r.src, "dst": r.dst, "amount": float(r.amount), "timestamp": int(r.timestamp),
                           "cross_border": int(r.cross_border), "importance": 0.5} for r in sub.itertuples()],
                "top_features": [{"feature": "near_thr_ratio", "weight": 0.4}, {"feature": "burst_6h", "weight": 0.2}]}
        facts = sar.build_facts(expl)
        sheet = sar.render_fact_sheet(facts)
        text = sar.template_narrative(facts)
        n_sent = len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()])
        toks = est_tokens(text)
        worst = max(worst, toks)
        ok_sent &= BRIEF["sar_sentences"][0] <= n_sent <= BRIEF["sar_sentences"][1]
        ok_tok &= toks < BRIEF["sar_tokens"]
        ok_det &= text == sar.template_narrative(sar.build_facts(expl))
        ok_val &= sar.validate_narrative(text, sheet)[0]
        ok_facts &= bool(re.search(r"\d{4}-\d{2}-\d{2}", text) and re.search(r"\$[\d,]+", text) and re.search(r"ACC\d+", text))
        detail.append(f"{typ}:{toks}")
    rec(S, ok_sent, "SAR narrative has 3-4 sentences (all 5 typologies)")
    rec(S, ok_tok, f"SAR narrative is under {BRIEF['sar_tokens']} tokens (estimate)", f"longest ~{worst} tokens ({', '.join(detail)})")
    rec(S, ok_det, "SAR narrative is deterministic (same input -> identical text)")
    rec(S, ok_val, "every number, date and account id in the narrative is traceable to the fact sheet (validator passes)")
    rec(S, ok_facts, "narrative contains specific date, dollar amount and account number")

    try:
        from xai_explainer.audit_log import AuditLog  # expected interface: AuditLog(path).append(dict) / .verify() -> bool
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "audit.log"
            log = AuditLog(str(p))
            for i in range(5):
                log.append({"event": "alert", "n": i})
            rec(S, bool(log.verify()), "audit log verifies after appending 5 events")
            raw = p.read_text()
            p.write_text(raw.replace('"n": 2', '"n": 9', 1) if '"n": 2' in raw else raw[:-2] + "x\n")
            rec(S, not AuditLog(str(p)).verify(), "audit log DETECTS tampering with an old entry (immutability)")
    except ImportError:
        skip(S, "audit log: append 5 events, verify, detect tampering", "needs xai_explainer/audit_log.py with AuditLog(path).append(dict) and .verify()")


# ================================================================================================= F
def section_f():
    S = "F"
    tests = glob.glob(str(ROOT / "tests" / "**" / "test_*.py"), recursive=True)
    rec(S, len(tests) > 0, "tests/ folder contains pytest tests", f"{len(tests)} test file(s)")
    for mod in ("aml_synth", "gnn_aml_core", "xai_explainer"):
        # Match by what the test file actually imports, not by filename substring: a sensibly-named test file
        # (test_graph_generator.py, testing aml_synth.graph_generator) need not contain the package name at all.
        found = False
        for t in tests:
            try:
                src = Path(t).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if re.search(rf"\b(from|import)\s+{re.escape(mod)}\b", src):
                found = True
                break
        rec(S, found, f"at least one test file for {mod}")
    try:
        import pytest  # noqa: F401
        import pytest_cov  # noqa: F401
    except ImportError:
        skip(S, f"coverage >= {BRIEF['coverage']}%", "pytest / pytest-cov not installed (pip install pytest pytest-cov)")
        return
    if not tests:
        rec(S, False, f"coverage >= {BRIEF['coverage']}%", "no tests to run")
        return
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--cov=aml_synth", "--cov=gnn_aml_core", "--cov=xai_explainer",
                          "--cov-report=term", "-x", "-p", "no:cacheprovider"], capture_output=True, text=True, timeout=900)
    m = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", out.stdout)
    rec(S, out.returncode == 0, "pytest passes", out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-200:])
    cov = int(m.group(1)) if m else 0
    rec(S, cov >= BRIEF["coverage"], f"coverage >= {BRIEF['coverage']}%", f"{cov}%")


# ================================================================================================= G
def section_g():
    S = "G"
    r = read("README.md")
    rec(S, "shields.io" in r or "badge" in r.lower(), "README has badges (PyPI, Zenodo DOI, licence, Hugging Face)")
    rec(S, "pip install xai-amlbench" in r, "README quick start: pip install xai-amlbench")
    rec(S, re.search(r"benchmark", r, re.I) is not None and "AUC" in r, "README has a performance benchmark table (vs. baselines)")
    rec(S, re.search(r"bibtex|@software|@misc|@article", r, re.I) is not None, "README has a citation guide (BibTeX)")
    rec(S, re.search(r"FinCEN", r) is not None, "README states the regulatory motivation (FinCEN)")
    for mod in ("aml_synth", "gnn_aml_core", "xai_explainer"):
        rec(S, (ROOT / mod / "README.md").exists(), f"{mod}/README.md with usage examples")


# ================================================================================================= main
def main():
    ap = argparse.ArgumentParser(description="XAI-AMLBench brief scorecard")
    ap.add_argument("--only", default="", help="comma list of sections, e.g. B,C")
    ap.add_argument("--api", default="", help="base URL of the running GNN API, e.g. http://localhost/gnn")
    ap.add_argument("--full", action="store_true", help="also generate a 50,000-node graph")
    ap.add_argument("--json", default="", help="write the results to this JSON file")
    ap.add_argument("--report-only", action="store_true", help="always exit 0")
    args = ap.parse_args()
    only = {s.strip().upper() for s in args.only.split(",") if s.strip()} or set(SECTIONS)

    runners = {"A": (section_a,), "B": (section_b, args.full), "C": (section_c,), "D": (section_d, args.api), "E": (section_e,),
               "F": (section_f,), "G": (section_g,)}
    t0 = time.time()
    print("XAI-AMLBench: brief scorecard\n" + "=" * 78)
    for sec in "ABCDEFG":
        if sec not in only:
            continue
        start = len(RESULTS)
        fn, *rest = runners[sec]
        guarded(sec, fn, *rest)
        print(f"\n{sec}  {SECTIONS[sec]}")
        for s, status, name, detail in RESULTS[start:]:
            mark = {"PASS": "  PASS", "FAIL": "  FAIL", "SKIP": "  skip"}[status]
            print(f"{mark}  {name}" + (f"\n          {detail}" if detail and status != "PASS" else (f"  [{detail}]" if detail else "")))

    p = sum(1 for r in RESULTS if r[1] == "PASS")
    f = sum(1 for r in RESULTS if r[1] == "FAIL")
    s = sum(1 for r in RESULTS if r[1] == "SKIP")
    print("\n" + "=" * 78)
    for sec in "ABCDEFG":
        rows = [r for r in RESULTS if r[0] == sec]
        if rows:
            print(f"  {sec}  {SECTIONS[sec]:<52} {sum(r[1] == 'PASS' for r in rows):>2}/{sum(r[1] != 'SKIP' for r in rows):<2} passed")
    total = p + f
    print(f"\n  OVERALL: {p}/{total} checks pass ({100 * p / max(total, 1):.0f}%), {f} fail, {s} skipped   ({time.time() - t0:.0f}s)")
    if args.json:
        Path(args.json).write_text(json.dumps({"passed": p, "failed": f, "skipped": s, "checks": [dict(section=a, status=b, name=c, detail=d) for a, b, c, d in RESULTS]}, indent=2))
        print(f"  saved {args.json}")
    return 0 if (args.report_only or f == 0) else 1


if __name__ == "__main__":
    sys.exit(main())