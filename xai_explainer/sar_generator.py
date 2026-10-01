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
    "dormant_reactivation": "sudden reactivation of a dormant account",
    "asymmetric_structuring": "disproportionate consolidation of many small inflows",
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
    "kyc_risk": "an elevated customer risk rating",
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

    # Checked first: cross-border velocity has the most specific signature (one subject, many transactions,
    # concentrated in a short window, spread across several countries), so it is checked before the shell-company
    # and cyclic-loop checks below, which can otherwise fire on it by coincidence -- a handful of counterparties
    # drawn from the general population will occasionally include 2+ shell accounts, and a subject that both pays
    # and is paid by the same counterparty (common in bidirectional cross-border activity) trivially forms a
    # 2-node cycle that is not the circular-fund-flow pattern the cyclic_loop check is meant to catch.
    subj_edges = [e for e in edges if subject in (e["src"], e["dst"])]
    if len(subj_edges) >= 10:
        ts = sorted(e["timestamp"] for e in subj_edges)
        cps = {e["dst"] if e["src"] == subject else e["src"] for e in subj_edges}
        out_n = sum(1 for e in subj_edges if e["src"] == subject)
        in_n = len(subj_edges) - out_n
        # cross-border velocity is bidirectional by construction (the subject both sends and receives, roughly
        # evenly), which tells it apart from a scatter_gather origin (pure fan-out, in_n == 0) or a smurfing hub
        # (fan-in from dozens to thousands of mules, with only 1-3 outgoing transfers to beneficiaries -- an
        # absolute floor like "at least 3" is not enough to rule that out, since 3 beneficiaries alone would pass
        # it on a large hub, so the minority direction must also be a real share of the traffic, not a corner case).
        minority = min(out_n, in_n)
        bidirectional = minority >= 3 and minority / len(subj_edges) >= 0.15
        max_amt = max(e["amount"] for e in subj_edges)
        # cross_border_velocity's own transfer amounts are capped at $9,500 by construction; dormant_reactivation's
        # can reach $150,000. A max amount well above that ceiling is a safe, well-justified exclusion -- found
        # by testing across many seeds, where a dormant_reactivation instance with high-value transactions and
        # a handful of foreign counterparties otherwise satisfied this check's other conditions too.
        if bidirectional and max_amt <= 10_500 and ts[-1] - ts[0] <= 5 * 24 * 3600 and len({countries.get(c) for c in cps} - {None}) >= 3:  # generator's velocity window is up to 96h
            return "cross_border_velocity"

    # Checked next, before shell_company and cyclic_loop: fan-in structuring (many distinct senders feeding one
    # hub account). Checked this early because a large, randomly-drawn counterparty pool (asymmetric_structuring
    # draws 15-45 general-population counterparties) will occasionally include 2+ shell-type accounts purely by
    # chance, which would otherwise trigger the shell_company check below on a pattern that structurally has
    # nothing to do with shell layering -- a real false positive found by testing across many seeds.
    near = sum(1 for e in edges if 0.8 * threshold <= e["amount"] < threshold)
    in_by_dst: dict[str, set[str]] = {}
    for e in edges:
        in_by_dst.setdefault(e["dst"], set()).add(e["src"])
    max_fan_in = max((len(s) for s in in_by_dst.values()), default=0)
    if in_by_dst:
        hub, senders = max(in_by_dst.items(), key=lambda kv: len(kv[1]))
        hub_edges = sum(1 for e in edges if e["dst"] == hub)
        hub_in_amts = [e["amount"] for e in edges if e["dst"] == hub]
        hub_out_amts = [e["amount"] for e in edges if e["src"] == hub]
        fan_in_shape = hub_edges / len(edges) >= 0.5
        # Sender count alone is not a reliable smurfing/asymmetric_structuring boundary: smurfing's own mule
        # count is capped relative to graph size (max(20, 0.04*n_accounts) in the generator), so on a smaller
        # graph it can land well below what looks like "obviously smurfing-scale" -- a real case found by
        # testing the exact configuration this project's own test suite uses (1,200 accounts), where a genuine
        # smurfing instance had only 48 distinct mules. The amount ceiling is the reliable signal instead:
        # smurfing deposits are capped near the reporting threshold (up to ~$9,900 by construction), while
        # asymmetric_structuring's inbound amounts are capped much lower (~$4,500) -- so within the shared
        # "large fan-in with one dominant payout" shape, the amount range decides which typology it is.
        if fan_in_shape and len(senders) >= 12 and hub_in_amts and hub_out_amts:
            large_payout = max(hub_out_amts) >= 3 * (sum(hub_in_amts) / len(hub_in_amts))
            if large_payout:
                if max(hub_in_amts) < 6000 and len(senders) <= 50:
                    return "asymmetric_structuring"
                return "smurfing"
    if near >= 5 and near / len(edges) >= 0.5 and max_fan_in >= 3:      # structuring: most transfers sit
        # just under the threshold, AND at least 3 distinct senders feed the SAME destination -- a single chain
        # transfer or cyclic flow trivially has multiple distinct senders across its different hops (each hop
        # sends once), but never many senders into the same one account, which is genuine smurfing's actual
        # shape. Checking distinct senders anywhere in the edge set (rather than fan-in to one destination) was
        # a real first attempt at this fix that did not work, caught by re-measuring after applying it.
        return "smurfing"

    if sum(1 for t in types.values() if t == "shell") >= 2:
        # Precise fix (not a reorder -- the reorder attempt above broke 288 other cases): require at least 2
        # shell-type accounts CONNECTED to each other in sequence (each both receiving from and sending to
        # another account in the subgraph), not just present anywhere. A randomly-drawn counterparty pool can
        # incidentally include 2+ shell-type accounts that are each simple one-off counterparties to the
        # subject, unconnected to each other -- that is not shell layering, which is specifically a CHAIN of
        # shells passing money to one another (see _shell_company's origin -> shell -> shell -> ... ->
        # beneficiary construction). This distinguishes the two without touching check order at all.
        #
        # "Connected" means sharing a direct edge with ANOTHER shell account -- not, as an earlier version of
        # this fix required, each shell independently having both an incoming AND an outgoing edge. That
        # stricter version broke a real, valid 2-shell chain ending directly at the final shell (no beneficiary
        # node after it): the last shell in such a chain has no outgoing edge at all, so it never satisfied
        # "out_degree >= 1" on its own, even though it is genuinely connected to the shell before it. Found by
        # an existing test failing after that fix shipped, not caught before.
        shell_accounts = {aid for aid, t in types.items() if t == "shell"}
        connected_shells = sum(
            1 for s in shell_accounts
            if any(other in shell_accounts and (g.has_edge(s, other) or g.has_edge(other, s))
                  for other in shell_accounts if other != s)
        )
        if connected_shells >= 2:
            return "shell_company"
    if _has_cycle(g):
        return "cyclic_loop"
    # dormant_reactivation: a genuinely old account (per generator construction, opened 700-3500 days before
    # reactivating) suddenly showing a short, elevated burst of activity. Uses real account age when the
    # explanation carries it (age_days on each node, threaded through from train.py's saved graph via
    # extractor.py -- see extractor.py's node construction) and falls back to a purely structural approximation
    # (few edges, short span, elevated amount, genuine bidirectionality) when it does not -- an older saved
    # model, or a non-aml_synth dataset such as Elliptic, will not have this field, and this must not crash.
    #
    # Checked AFTER shell_company/cyclic_loop, not before: an earlier attempt moved this check first (reasoning:
    # incidental shell-type counterparties can trigger shell_company's crude ">=2 shells" test on a
    # dormant_reactivation instance by chance). That reorder was tried and measured -- it fixed 6 cases in that
    # direction but broke 288 in the other (real shell_company and cyclic_loop instances misclassified as
    # dormant_reactivation instead, since their own small, sometimes-bidirectional shape satisfies this check's
    # conditions too easily once checked first). Reverted; the smaller, remaining incidental-shell collision is
    # accepted rather than trading it for a much larger one.
    ages = {n["account_id"]: n.get("age_days") for n in nodes}
    if subj_edges:
        ts_all = sorted(e["timestamp"] for e in subj_edges)
        span_h = (ts_all[-1] - ts_all[0]) / 3600
        mean_amt = sum(e["amount"] for e in subj_edges) / len(subj_edges)
        out_n = sum(1 for e in subj_edges if e["src"] == subject)
        in_n = len(subj_edges) - out_n
        # requiring a genuine mix of both directions (not just "few edges, short span, high amount") is
        # essential: scatter_gather's origin node is pure fan-out (all outbound, never inbound) and otherwise
        # matches this shape closely enough to be misclassified as dormant_reactivation without this check --
        # a real regression found by testing across many seeds, not a hypothetical edge case.
        # min(in_n, out_n) >= 1 (not >= 2): with as few as 4 total transactions by construction, an even 2-2
        # split is not guaranteed even for a genuine instance -- >= 2 was too strict and rejected real cases by
        # chance (a 1-in/3-out split, say). >= 1 still excludes scatter_gather's pure fan-out origin (in_n == 0
        # always), which is the actual collision this check exists to prevent.
        reactivation_burst = len(subj_edges) <= 16 and span_h <= 72 and mean_amt >= 1200 and min(in_n, out_n) >= 1
        subject_age = ages.get(subject)
        if subject_age is not None:
            # real age available: require genuine dormancy (365 days is a full year below the generator's
            # 700-day floor, a safety margin) AND the reactivation shape -- age alone is not enough, since an
            # old account can also transact normally; the burst shape confirms reactivation, not just age.
            if subject_age >= 365 and reactivation_burst:
                return "dormant_reactivation"
        elif reactivation_burst:
            return "dormant_reactivation"
    for n in g.nodes:
        succ = set(g.successors(n))
        if len(succ) >= 4:
            sinks = [set(g.successors(s)) for s in succ if g.out_degree(s)]
            if sinks and max(sum(1 for s in sinks if t in s) for t in set().union(*sinks)) >= max(3, len(succ) // 2):
                return "scatter_gather"
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
    s1 = (f"Account {f['subject']} was flagged (risk score {f['score_pct']}) and is linked to "
          f"{f['n_accounts']} accounts in activity consistent with {f['typology_label']}.")
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
    elif t == "dormant_reactivation":
        s3 = ("The account showed little prior activity before a short burst of unusually large transactions, "
              "consistent with a dormant account being reactivated.")
    elif t == "asymmetric_structuring":
        s3 = (f"{f['n_tx']} smaller inbound transfers were consolidated into a single disproportionately large "
              "outbound payment, consistent with structured fund consolidation.")
    else:
        s3 = f"{f['n_cross']} of the {f['n_tx']} transactions were cross-border and the pattern deviates from expected account behavior."
    ind = _join(f["indicators"]) if f["indicators"] else "the overall transaction structure"
    s4 = (f"Key indicators include {ind}, and the activity is referred for compliance review "
          "and possible SAR filing.")
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
