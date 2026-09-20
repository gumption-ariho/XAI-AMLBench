"""Deterministic SAR-narrative pipeline: explanation -> facts -> fact sheet -> (LLM | template) -> validation.

Design goal: the narrative can never contain a number, date, amount or account id that is not in the
fact sheet. The LLM (if enabled) only paraphrases the fact sheet; `validate_narrative` rejects anything
else and the caller falls back to `template_narrative`, which is fully deterministic.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

import networkx as nx

TYPOLOGY_LABELS = {
    "smurfing": "structuring (smurfing)",
    "scatter_gather": "scatter-gather layering",
    "cyclic_loop": "circular fund flow",
    "shell_company": "layering through shell entities",
    "cross_border_velocity": "high-velocity cross-border activity",
    "unclassified": "an unclassified anomalous pattern",
}

# feature name (gnn_aml_core.features.FEATURE_NAMES) -> plain-language indicator (no numbers!)
INDICATOR_PHRASES = {
    "near_thr_ratio": "a high share of transfers just below the reporting threshold",
    "near_thr_cnt": "a high share of transfers just below the reporting threshold",
    "xb_out_ratio": "an elevated share of cross-border transfers",
    "xb_in_ratio": "an elevated share of cross-border transfers",
    "n_cp_countries": "counterparties spread across many jurisdictions",
    "burst_6h": "an unusual burst of transactions within a short window",
    "flow_ratio": "funds passing through with little balance retained",
    "retained_frac": "funds passing through with little balance retained",
    "out_uniq": "unusual counterparty fan-out",
    "in_uniq": "unusual counterparty fan-in",
    "out_deg": "unusual counterparty fan-out",
    "in_deg": "unusual counterparty fan-in",
    "type_shell": "shell or offshore account characteristics",
    "is_offshore": "shell or offshore account characteristics",
    "age_days": "a recently opened account",
    "active_span_h": "a compressed period of activity",
}
_BANNED = ("guilty", "convicted", "criminal", "terrorist", "proves", "definitely")


# ------------------------------------------------------------------ typology
def infer_typology(subject: str, nodes: list[dict], edges: list[dict], threshold: float) -> str:
    """Transparent structural heuristics over the explained subgraph (not a second model)."""
    if not edges:
        return "unclassified"
    g = nx.MultiDiGraph()
    for e in edges:
        g.add_edge(e["src"], e["dst"], amount=e["amount"], ts=e["timestamp"], xb=e["cross_border"])
    types = {n["account_id"]: n["account_type"] for n in nodes}
    countries = {n["account_id"]: n["country"] for n in nodes}

    if sum(1 for t in types.values() if t == "shell") >= 2:
        return "shell_company"
    if _has_cycle(g):
        return "cyclic_loop"
    near = sum(1 for e in edges if 0.8 * threshold <= e["amount"] < threshold)
    if near >= 5 and near / len(edges) >= 0.5:      # structuring: most transfers sit just under the threshold
        return "smurfing"
    for n in g.nodes:
        succ = set(g.successors(n))
        if len(succ) >= 4:
            sinks = [set(g.successors(s)) for s in succ if g.out_degree(s)]
            if sinks and max(sum(1 for s in sinks if t in s) for t in set().union(*sinks)) >= max(3, len(succ) // 2):
                return "scatter_gather"
    subj_edges = [e for e in edges if subject in (e["src"], e["dst"])]
    if len(subj_edges) >= 10:
        ts = sorted(e["timestamp"] for e in subj_edges)
        cps = {e["dst"] if e["src"] == subject else e["src"] for e in subj_edges}
        if ts[-1] - ts[0] <= 12 * 3600 and len({countries.get(c) for c in cps} - {None}) >= 3:
            return "cross_border_velocity"
    return "unclassified"


def _has_cycle(g: nx.MultiDiGraph) -> bool:
    try:
        nx.find_cycle(nx.DiGraph(g))
        return True
    except nx.NetworkXNoCycle:
        return False


# --------------------------------------------------------------------- facts
def _d(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def build_facts(explanation: dict) -> dict:
    """Reduce a /explain response to a small dict of verified, pre-formatted facts."""
    edges, nodes = explanation["edges"], explanation["nodes"]
    thr = float(explanation.get("reporting_threshold", 10_000))
    subject = explanation["account_id"]
    by_id = {n["account_id"]: n for n in nodes}
    subj = by_id.get(subject, {"account_type": "unknown", "country": "unknown"})

    typology = infer_typology(subject, nodes, edges, thr)
    ts = [e["timestamp"] for e in edges] or [0]
    span_h = max(1, round((max(ts) - min(ts)) / 3600))
    total = sum(e["amount"] for e in edges)
    countries = sorted({n["country"] for n in nodes})
    indicators: list[str] = []
    for f in explanation.get("top_features", []):
        phrase = INDICATOR_PHRASES.get(f["feature"])
        if phrase and phrase not in indicators:
            indicators.append(phrase)

    return {
        "subject": subject,
        "subject_type": subj["account_type"],
        "subject_country": subj["country"],
        "score_pct": f"{round(100 * explanation['risk_score'])}%",
        "typology": typology,
        "typology_label": TYPOLOGY_LABELS[typology],
        "n_accounts": str(len(nodes)),
        "n_tx": str(len(edges)),
        "total": f"${total:,.0f}",
        "start": _d(min(ts)),
        "end": _d(max(ts)),
        "span_h": str(span_h),
        "n_cross": str(sum(1 for e in edges if e["cross_border"])),
        "near_thr": str(sum(1 for e in edges if 0.8 * thr <= e["amount"] < thr)),
        "threshold": f"${thr:,.0f}",
        "shells": str(sum(1 for n in nodes if n["account_type"] == "shell")),
        "countries": countries,
        "n_countries": str(len(countries)),
        "indicators": indicators[:3],
    }


def render_fact_sheet(f: dict) -> str:
    lines = [
        f"SUBJECT ACCOUNT: {f['subject']} ({f['subject_type']}, country {f['subject_country']})",
        f"MODEL RISK SCORE: {f['score_pct']}",
        f"PATTERN (heuristic): {f['typology_label']}",
        f"ACCOUNTS INVOLVED: {f['n_accounts']}",
        f"TRANSACTIONS REVIEWED: {f['n_tx']}",
        f"TOTAL VALUE: {f['total']}",
        f"ACTIVITY WINDOW: {f['start']} to {f['end']} (about {f['span_h']} hours)",
        f"CROSS-BORDER TRANSACTIONS: {f['n_cross']} of {f['n_tx']}",
        f"TRANSACTIONS JUST BELOW THE {f['threshold']} REPORTING THRESHOLD: {f['near_thr']}",
        f"SHELL ENTITIES: {f['shells']}",
        f"COUNTRIES ({f['n_countries']}): {', '.join(f['countries'])}",
        "KEY MODEL INDICATORS: " + ("; ".join(f["indicators"]) if f["indicators"] else "none reported"),
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------- narrative
def template_narrative(f: dict) -> str:
    """Fully deterministic 4-sentence fallback (also the reference style for the LLM)."""
    s1 = (f"Account {f['subject']} was flagged by the graph model with a risk score of {f['score_pct']} "
          f"and is connected to {f['n_accounts']} accounts in activity consistent with {f['typology_label']}.")
    when = f"on {f['start']}" if f["start"] == f["end"] else f"between {f['start']} and {f['end']}"
    s2 = (f"The reviewed network contains {f['n_tx']} transactions totaling {f['total']}, "
          f"occurring {when} over about {f['span_h']} hours.")
    t = f["typology"]
    if t == "smurfing":
        s3 = (f"{f['near_thr']} of these transfers fell just below the {f['threshold']} reporting threshold, "
              "which is consistent with structuring.")
    elif t == "shell_company":
        s3 = (f"{f['shells']} of the involved accounts are shell entities through which funds were passed "
              "in a layered chain.")
    elif t == "cyclic_loop":
        s3 = "Funds returned to the originating accounts through a closed loop, consistent with circular movement."
    elif t == "scatter_gather":
        s3 = ("Funds fanned out from a single source to several intermediaries and were then consolidated "
              "into one destination account.")
    elif t == "cross_border_velocity":
        s3 = (f"The activity spans {f['n_countries']} jurisdictions ({', '.join(f['countries'])}) within a short "
              "window, indicating rapid cross-border movement of funds.")
    else:
        s3 = f"{f['n_cross']} of the {f['n_tx']} transactions were cross-border and the pattern deviates from expected account behavior."
    ind = _join(f["indicators"]) if f["indicators"] else "the overall transaction structure"
    s4 = (f"Key model indicators include {ind}, and the activity is referred for compliance officer review "
          "and potential Suspicious Activity Report filing.")
    return " ".join([s1, s2, s3, s4])


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_messages(fact_sheet: str) -> list[dict]:
    system = (
        "You are a compliance analyst assistant drafting the narrative section of a Suspicious Activity Report (SAR). "
        "Use ONLY the facts in the FACT SHEET. Write exactly 4 sentences of plain prose: no bullet points, no headings, "
        "no preamble. Copy every number, date, dollar amount and account ID exactly as written in the fact sheet. "
        "Use hedged language such as 'consistent with' or 'suspected'. Do not speculate about identities, motives or "
        "facts that are not listed."
    )
    user = f"FACT SHEET\n{fact_sheet}\n\nWrite the 4-sentence SAR narrative now."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------- validation
_NUM = re.compile(r"\d[\d,]*\.?\d*")
_ID = re.compile(r"\b[A-Z]{2,}\d{4,}\b")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _numbers(text: str) -> set[str]:
    return {m.rstrip(".").replace(",", "") for m in _NUM.findall(text)}


def validate_narrative(text: str, fact_sheet: str) -> tuple[bool, list[str]]:
    problems: list[str] = []
    text = text.strip()
    n_sent = len([s for s in _SENT.split(text) if s.strip()])
    if not 3 <= n_sent <= 4:
        problems.append(f"expected 3-4 sentences, got {n_sent}")
    extra_nums = _numbers(text) - _numbers(fact_sheet)
    if extra_nums:
        problems.append(f"numbers not in fact sheet: {sorted(extra_nums)}")
    extra_ids = set(_ID.findall(text)) - set(_ID.findall(fact_sheet))
    if extra_ids:
        problems.append(f"account ids not in fact sheet: {sorted(extra_ids)}")
    low = text.lower()
    if any(w in low for w in _BANNED):
        problems.append("contains conclusive/accusatory language")
    if re.search(r"^\s*([-*•]|\d+[.)])\s", text, flags=re.M) or "\n\n" in text:
        problems.append("contains list/paragraph formatting")
    return (not problems), problems


def narrative_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
