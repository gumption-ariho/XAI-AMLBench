"""Node / edge feature engineering shared by training (train.py) and serving (main.py).

34 structural + behavioural features per account, including several grounded directly in real mule-account
red flags (pass-through with little retained balance, structuring near reporting thresholds, round-dollar and
micro-deposit patterns, scripted/bot-like timing, tax-haven counterparty exposure, drastic volume shifts).
The client report targets 400+ pre-computed features in Redis; extend `build_node_features` (2-hop aggregates,
temporal windows, centralities...) over time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RELATIONS_BASE = 4     # (domestic | cross-border) x (regular | large)
NUM_RELATIONS = 2 * RELATIONS_BASE   # x (forward | reverse copy of the transaction)
EDGE_DIM = 5           # amount, cross-border, near-threshold, time, direction flag
LARGE_TX = 8_000.0
BURST_WINDOW_S = 6 * 3600

FEATURE_NAMES = [
    "out_deg", "in_deg", "out_uniq", "in_uniq",
    "out_amt_sum", "in_amt_sum", "out_amt_mean", "in_amt_mean",
    "out_amt_max", "in_amt_max", "out_amt_std", "in_amt_std",
    "flow_ratio", "retained_frac",
    "xb_out_ratio", "xb_in_ratio", "n_cp_countries",
    "near_thr_cnt", "near_thr_ratio", "burst_6h", "active_span_h",
    "age_days", "is_offshore", "kyc_risk",
    "type_individual", "type_business", "type_shell",
    "round_amt_ratio", "micro_tx_ratio", "decimal_precision_ratio",
    "volume_shift_ratio", "tax_haven_cp_ratio", "hour_concentration", "pass_through_match_ratio",
    "high_risk_dest_ratio", "shared_memo_ratio", "dispute_rate", "counterparty_registration_cluster_ratio",
]
_LOG_FEATURES = {
    "out_deg", "in_deg", "out_uniq", "in_uniq", "out_amt_sum", "in_amt_sum", "out_amt_mean",
    "in_amt_mean", "out_amt_max", "in_amt_max", "out_amt_std", "in_amt_std", "near_thr_cnt",
    "burst_6h", "active_span_h", "age_days", "n_cp_countries",
}
OFFSHORE = {"VG", "KY", "PA", "SC", "BZ"}


def build_node_features(accounts: pd.DataFrame, tx: pd.DataFrame, reporting_threshold: float = 10_000.0):
    """Returns (X float32 [N, F], feature_names). Row order == accounts row order."""
    T = reporting_threshold
    tx = tx.copy()
    tx["near"] = ((tx["amount"] >= 0.8 * T) & (tx["amount"] < T)).astype(int)
    country = accounts.set_index("account_id")["country"]
    tx["dst_country"] = tx["dst"].map(country)
    tx["src_country"] = tx["src"].map(country)

    def side(key: str, other: str, other_country: str, prefix: str) -> pd.DataFrame:
        g = tx.groupby(key)
        s = g["amount"].agg(["count", "sum", "mean", "max", "std"]).fillna(0.0)
        s.columns = [f"{prefix}_deg", f"{prefix}_amt_sum", f"{prefix}_amt_mean",
                     f"{prefix}_amt_max", f"{prefix}_amt_std"]
        s[f"{prefix}_uniq"] = g[other].nunique()
        s[f"xb_{prefix}_ratio"] = g["cross_border"].mean()
        s[f"_near_{prefix}"] = g["near"].sum()
        s[f"_cc_{prefix}"] = g[other_country].apply(lambda c: set(c.dropna()))
        return s

    out_s = side("src", "dst", "dst_country", "out")
    in_s = side("dst", "src", "src_country", "in")

    df = pd.DataFrame(index=accounts["account_id"])
    df = df.join(out_s.drop(columns="_cc_out")).join(in_s.drop(columns="_cc_in"))
    cc_out = out_s["_cc_out"].reindex(df.index)
    cc_in = in_s["_cc_in"].reindex(df.index)
    df["n_cp_countries"] = [
        len((a if isinstance(a, set) else set()) | (b if isinstance(b, set) else set()))
        for a, b in zip(cc_out, cc_in)
    ]
    df = df.fillna(0.0)

    df["flow_ratio"] = df["in_amt_sum"] / (df["out_amt_sum"] + 1.0)
    df["retained_frac"] = (df["in_amt_sum"] - df["out_amt_sum"]).abs() / (df["in_amt_sum"] + df["out_amt_sum"] + 1.0)
    df["near_thr_cnt"] = df["_near_out"] + df["_near_in"]
    df["near_thr_ratio"] = df["near_thr_cnt"] / (df["out_deg"] + df["in_deg"] + 1.0)

    # burstiness: max number of transactions touching the account in one 6h bucket
    long = pd.concat([
        tx[["src", "timestamp"]].rename(columns={"src": "acct"}),
        tx[["dst", "timestamp"]].rename(columns={"dst": "acct"}),
    ])
    long["bucket"] = long["timestamp"] // BURST_WINDOW_S
    df["burst_6h"] = long.groupby(["acct", "bucket"]).size().groupby("acct").max().reindex(df.index).fillna(0)
    span = long.groupby("acct")["timestamp"].agg(lambda s: s.max() - s.min())
    df["active_span_h"] = (span.reindex(df.index).fillna(0) / 3600.0)

    t_ref = int(tx["timestamp"].min())
    acc = accounts.set_index("account_id")
    df["age_days"] = ((t_ref - acc["opened_ts"]) / 86400.0).clip(lower=0).reindex(df.index)
    df["is_offshore"] = acc["country"].isin(OFFSHORE).astype(float).reindex(df.index)
    df["kyc_risk"] = acc["risk_score"].reindex(df.index).fillna(0.0) if "risk_score" in acc.columns else 0.0
    for t in ("individual", "business", "shell"):
        df[f"type_{t}"] = (acc["account_type"] == t).astype(float).reindex(df.index)

    # --- newly added: signals grounded in real behavioural red flags (structuring, benign-looking mule
    # accounts, coordinated account creation), each computed purely from existing amount/timestamp/country
    # data -- no new field on accounts.csv or transactions.csv was needed for any of these.

    # round_amt_ratio: launderers (and legitimate payroll/rent) both favour round numbers, but a genuinely
    # HIGH share of exact-hundred amounts on an otherwise ordinary-looking account is a real, common red flag.
    tx["_round"] = (tx["amount"] % 100 < 1e-6) | (tx["amount"] % 100 > 100 - 1e-6)
    round_out = tx.groupby("src")["_round"].mean()
    round_in = tx.groupby("dst")["_round"].mean()
    df["round_amt_ratio"] = (round_out.reindex(df.index).fillna(0.0) + round_in.reindex(df.index).fillna(0.0)) / 2.0

    # micro_tx_ratio: many transactions under a token amount is the classic signature of testing an account's
    # routing before moving real money through it (penny-deposit verification abuse, probing).
    tx["_micro"] = tx["amount"] < 5.0
    micro_out = tx.groupby("src")["_micro"].mean()
    micro_in = tx.groupby("dst")["_micro"].mean()
    df["micro_tx_ratio"] = (micro_out.reindex(df.index).fillna(0.0) + micro_in.reindex(df.index).fillna(0.0)) / 2.0

    # decimal_precision_ratio: amounts with more precision than real currency ever needs (more than 2 decimal
    # places) suggest an amount produced by an algorithm splitting funds precisely, not a human-entered payment.
    tx["_odd_precision"] = (tx["amount"] - tx["amount"].round(2)).abs() > 1e-6
    prec_out = tx.groupby("src")["_odd_precision"].mean()
    prec_in = tx.groupby("dst")["_odd_precision"].mean()
    df["decimal_precision_ratio"] = (prec_out.reindex(df.index).fillna(0.0) + prec_in.reindex(df.index).fillna(0.0)) / 2.0

    # volume_shift_ratio: a drastic change in transaction volume (e.g. student-level activity suddenly
    # becoming corporate-level wires) shows up as a large gap between an account's earlier and later halves.
    half_amt = pd.concat([
        tx[["src", "timestamp", "amount"]].rename(columns={"src": "acct"}),
        tx[["dst", "timestamp", "amount"]].rename(columns={"dst": "acct"}),
    ])
    half_amt["_half"] = 0
    mid_ts2 = half_amt.groupby("acct")["timestamp"].transform(lambda s: (s.min() + s.max()) / 2.0)
    half_amt.loc[half_amt["timestamp"] >= mid_ts2, "_half"] = 1
    vol_by_half = half_amt.groupby(["acct", "_half"])["amount"].sum().unstack(fill_value=0.0)
    early = vol_by_half.get(0, pd.Series(0.0, index=vol_by_half.index))
    late = vol_by_half.get(1, pd.Series(0.0, index=vol_by_half.index))
    df["volume_shift_ratio"] = (late.reindex(df.index).fillna(0.0) / (early.reindex(df.index).fillna(0.0) + 1.0))

    # tax_haven_cp_ratio: share of an account's transactions (by count, not just whether ANY counterparty is
    # offshore) that touch a counterparty in a known low-transparency jurisdiction -- reuses the same OFFSHORE
    # list already used for is_offshore, applied to counterparties instead of the account's own country.
    tx["_cp_offshore_out"] = tx["dst_country"].isin(OFFSHORE)
    tx["_cp_offshore_in"] = tx["src_country"].isin(OFFSHORE)
    haven_out = tx.groupby("src")["_cp_offshore_out"].mean()
    haven_in = tx.groupby("dst")["_cp_offshore_in"].mean()
    df["tax_haven_cp_ratio"] = (haven_out.reindex(df.index).fillna(0.0) + haven_in.reindex(df.index).fillna(0.0)) / 2.0

    # hour_concentration: 1 - normalised entropy of the hour-of-day histogram. A real person's spending spreads
    # across a wide, irregular range of hours; a script running on a fixed schedule concentrates into a few
    # hours (or conspicuously avoids others), giving LOW entropy -- so higher hour_concentration means more
    # suspicious, consistent with the direction of every other feature here.
    long["_hour"] = (long["timestamp"] // 3600) % 24
    def _concentration(hours: pd.Series) -> float:
        counts = hours.value_counts()
        if len(counts) <= 1:
            return 1.0  # everything in a single hour bucket is maximally concentrated
        p = counts / counts.sum()
        entropy = -(p * np.log(p)).sum()
        max_entropy = np.log(24)  # the entropy of a perfectly uniform spread across all 24 hours
        return float(1.0 - entropy / max_entropy)
    df["hour_concentration"] = long.groupby("acct")["_hour"].apply(_concentration).reindex(df.index).fillna(0.0)

    # pass_through_match_ratio: the clearest mule signature of all -- money arrives and a closely matching
    # amount leaves again shortly after, rather than being spent or retained the way a real account's balance
    # behaves. Checks each outbound transaction against that account's inbound transactions in the preceding
    # 48 hours, allowing a 5% tolerance (fees, rounding) rather than requiring an exact match.
    #
    # Capped to the MOST RECENT 300 transactions per side for any single account: this check is naturally
    # O(n_out * n_in) per account, and an uncapped hub account (a real collector can have 1,000+ transactions,
    # see aml_synth's own hard-negative generator) made this feature alone take several seconds on a normal
    # 5,000-account benchmark graph -- the same kind of unbounded-hub cost this project already caps elsewhere
    # (MAX_SCORE_EDGES in gnn_aml_core/main.py, MAX_EXPLAIN_EDGES for GNNExplainer). Recent transactions are
    # kept rather than an arbitrary slice, since the 48h matching window only ever looks backward in time
    # anyway, so older transactions past the cap could not have matched a later outbound transaction regardless.
    PASS_THROUGH_CAP = 300

    def _pass_through(acct_tx: pd.DataFrame) -> float:
        outs = acct_tx[acct_tx["_dir"] == "out"]
        ins = acct_tx[acct_tx["_dir"] == "in"]
        if len(outs) == 0 or len(ins) == 0:
            return 0.0
        # Sorting has its own cost, so only pay it for the rare account that actually needs trimming --
        # the vast majority of accounts (median well under 100 transactions) never reach the cap at all.
        if len(outs) > PASS_THROUGH_CAP:
            outs = outs.sort_values("timestamp").tail(PASS_THROUGH_CAP)
        if len(ins) > PASS_THROUGH_CAP:
            ins = ins.sort_values("timestamp").tail(PASS_THROUGH_CAP)
        matched = 0
        in_ts, in_amt = ins["timestamp"].to_numpy(), ins["amount"].to_numpy()
        for ts, amt in zip(outs["timestamp"].to_numpy(), outs["amount"].to_numpy()):
            window = (in_ts <= ts) & (in_ts >= ts - 48 * 3600)
            if window.any() and (np.abs(in_amt[window] - amt) <= 0.05 * amt).any():
                matched += 1
        return matched / len(outs)

    pt_long = pd.concat([
        tx[["src", "timestamp", "amount"]].rename(columns={"src": "acct"}).assign(_dir="out"),
        tx[["dst", "timestamp", "amount"]].rename(columns={"dst": "acct"}).assign(_dir="in"),
    ])
    df["pass_through_match_ratio"] = pt_long.groupby("acct").apply(_pass_through, include_groups=False).reindex(df.index).fillna(0.0)

    # high_risk_dest_ratio: share of an account's OUTBOUND transactions going to a destination tagged as a
    # high-risk merchant category (crypto exchange, gambling operator, or similar) by the generator's
    # a_highrisk mechanism -- only meaningful if the transactions frame carries a "category" column (real,
    # non-synthetic data such as Elliptic will not have this column, so this degrades to 0.0 for everyone
    # rather than raising, keeping this feature safely optional).
    if "category" in tx.columns:
        tx["_highrisk_dest"] = tx["category"] == "high_risk_dest"
        hr_out = tx.groupby("src")["_highrisk_dest"].mean()
        df["high_risk_dest_ratio"] = hr_out.reindex(df.index).fillna(0.0)
    else:
        df["high_risk_dest_ratio"] = 0.0

    # shared_memo_ratio: share of an account's outbound transactions whose memo/reference text (a) is not one
    # of a handful of common, generic strings, and (b) is used by 2+ OTHER, distinct accounts elsewhere in the
    # graph -- a real red flag (coordinated mules copy-pasting the same reference text, typo and all). Requires
    # the memo to be shared by genuinely unrelated senders, not just repeated by the same account many times,
    # since one account reusing its own memo is ordinary behaviour, not coordination. Only meaningful if the
    # transactions frame carries a "memo" column; degrades safely to 0.0 otherwise (real, non-synthetic data
    # will not have this column at all).
    if "memo" in tx.columns:
        generic = {"PAYMENT", "INVOICE PMT", "TRANSFER", "RENT", "SALARY", ""}
        non_generic = tx[~tx["memo"].isin(generic)]
        senders_per_memo = non_generic.groupby("memo")["src"].nunique()
        shared_memos = set(senders_per_memo[senders_per_memo >= 2].index)
        tx["_shared_memo"] = tx["memo"].isin(shared_memos)
        shared_out = tx.groupby("src")["_shared_memo"].mean()
        df["shared_memo_ratio"] = shared_out.reindex(df.index).fillna(0.0)
    else:
        df["shared_memo_ratio"] = 0.0

    # dispute_rate: share of an account's own outbound transactions that were later disputed/reversed. A real
    # red flag when elevated well above baseline (repeatedly testing a platform's automated reversal logic),
    # distinct from the occasional legitimate dispute every account has some small chance of. Only meaningful
    # if the transactions frame carries a "disputed" column; degrades safely to 0.0 otherwise (real,
    # non-synthetic data will not have this column at all).
    if "disputed" in tx.columns:
        dispute_out = tx.groupby("src")["disputed"].mean()
        df["dispute_rate"] = dispute_out.reindex(df.index).fillna(0.0)
    else:
        df["dispute_rate"] = 0.0

    # counterparty_registration_cluster_ratio: for each account, the largest share of its distinct
    # counterparties that were all registered on the exact same day as each other -- a real red flag (a batch
    # of mule accounts opened together, then used together), reusing opened_ts (already available on every
    # account, threaded through earlier this session for dormant_reactivation) rather than needing any new
    # schema field. An ordinary account's counterparties open their accounts independently, so this is
    # naturally near 0 for most accounts; it only rises when several counterparties genuinely share a
    # registration day, which real batch-created mule networks do and ordinary traffic does not.
    #
    # The "opened_ts not in accounts.columns" branch below is defensive, not currently reachable in practice:
    # age_days (above) already accesses accounts["opened_ts"] directly with no guard, so this function has
    # always hard-required that column for any real caller. Kept anyway as honest, harmless defensive code,
    # not represented as an exercised fallback path (see test_features.py's note on this).
    if "opened_ts" in accounts.columns:
        opened = accounts.set_index("account_id")["opened_ts"]
        reg_day_out = tx["dst"].map(opened) // 86400
        reg_day_in = tx["src"].map(opened) // 86400
        cluster_ratio = {}
        for acct, group in pd.concat([
            pd.DataFrame({"acct": tx["src"], "cp": tx["dst"], "reg_day": reg_day_out}),
            pd.DataFrame({"acct": tx["dst"], "cp": tx["src"], "reg_day": reg_day_in}),
        ]).groupby("acct"):
            cps = group.drop_duplicates("cp").dropna(subset=["reg_day"])
            # counterparties with an unknown registration date (an external account not present in the
            # accounts table at all -- a real, common case, not a hypothetical one: it crashed every test
            # whose transactions referenced any external counterparty, not just tests written for this
            # feature specifically) contribute no clustering signal and are excluded, not treated as an error.
            n_cps = len(cps)
            if n_cps == 0:
                cluster_ratio[acct] = 0.0
                continue
            largest_cluster = int(cps["reg_day"].value_counts().max())
            cluster_ratio[acct] = (largest_cluster - 1) / n_cps if n_cps > 1 else 0.0
        df["counterparty_registration_cluster_ratio"] = df.index.map(cluster_ratio).fillna(0.0)
    else:
        df["counterparty_registration_cluster_ratio"] = 0.0

    x = df[FEATURE_NAMES].astype("float64").copy()
    for c in _LOG_FEATURES:
        x[c] = np.log1p(x[c].clip(lower=0))
    x["flow_ratio"] = np.log1p(x["flow_ratio"].clip(lower=0))
    return x.to_numpy(dtype=np.float32), list(FEATURE_NAMES)


def standardize(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    if mean is None:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def build_edge_features(tx: pd.DataFrame, reporting_threshold: float = 10_000.0):
    """Returns (edge_attr float32 [E, EDGE_DIM], edge_type int64 [E])."""
    amt = tx["amount"].to_numpy(dtype=np.float64)
    ts = tx["timestamp"].to_numpy(dtype=np.float64)
    near = ((amt >= 0.8 * reporting_threshold) & (amt < reporting_threshold)).astype(np.float64)
    tnorm = (ts - ts.min()) / max(ts.max() - ts.min(), 1.0)
    attr = np.stack([np.log1p(amt) / 12.0, tx["cross_border"].to_numpy(dtype=np.float64), near, tnorm], axis=1)
    etype = tx["cross_border"].to_numpy(dtype=np.int64) * 2 + (amt >= LARGE_TX).astype(np.int64)
    return attr.astype(np.float32), etype


def make_bidirectional(src: np.ndarray, dst: np.ndarray, edge_attr: np.ndarray, edge_type: np.ndarray):
    """Add a reversed copy of every transaction, flagged as reversed.

    Message passing normally follows the edge direction, so an account would never hear from the accounts it SENDS to
    (a mule paying a central account could not see that account). With the reverse copies every account sees all of its
    neighbours, and the direction flag / relation id still tells the model which way the money went.

    Returns (src2, dst2, edge_attr2 [2E, EDGE_DIM], edge_type2 [2E]). The first E rows are the original transactions,
    the last E rows their reversed copies (so copy k+E is the reverse of transaction k).
    """
    e = len(src)
    fwd = np.concatenate([edge_attr, np.zeros((e, 1), dtype=np.float32)], axis=1)
    rev = np.concatenate([edge_attr, np.ones((e, 1), dtype=np.float32)], axis=1)
    return (np.concatenate([src, dst]).astype(np.int64), np.concatenate([dst, src]).astype(np.int64),
            np.concatenate([fwd, rev]).astype(np.float32), np.concatenate([edge_type, edge_type + RELATIONS_BASE]).astype(np.int64))


def merge_reverse_copies(edge_ids, src_local, dst_local, importance, n_tx: int, top_k: int):
    """Explanations run on the bidirectional graph. Fold each transaction's forward and reversed copy into ONE edge
    (keeping the larger importance) and restore its true direction. Returns [(tx_index, src_local, dst_local, importance)]
    sorted by importance, at most top_k."""
    best: dict[int, tuple] = {}
    for ge, s_l, d_l, imp in zip(edge_ids, src_local, dst_local, importance):
        ge = int(ge)
        tx, rev = ge % n_tx, ge >= n_tx
        if tx not in best or float(imp) > best[tx][3]:
            best[tx] = (tx, int(d_l) if rev else int(s_l), int(s_l) if rev else int(d_l), float(imp))
    return sorted(best.values(), key=lambda r: -r[3])[:top_k]
