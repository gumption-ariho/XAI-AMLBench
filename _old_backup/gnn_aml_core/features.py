"""Node / edge feature engineering shared by training (train.py) and serving (main.py).

This is a compact starter set (~26 structural + behavioural features per account).
The client report targets 400+ pre-computed features in Redis; extend
`build_node_features` (2-hop aggregates, temporal windows, centralities...) over time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUM_RELATIONS = 4      # (domestic | cross-border) x (regular | large)
EDGE_DIM = 4
LARGE_TX = 8_000.0
BURST_WINDOW_S = 6 * 3600

FEATURE_NAMES = [
    "out_deg", "in_deg", "out_uniq", "in_uniq",
    "out_amt_sum", "in_amt_sum", "out_amt_mean", "in_amt_mean",
    "out_amt_max", "in_amt_max", "out_amt_std", "in_amt_std",
    "flow_ratio", "retained_frac",
    "xb_out_ratio", "xb_in_ratio", "n_cp_countries",
    "near_thr_cnt", "near_thr_ratio", "burst_6h", "active_span_h",
    "age_days", "is_offshore",
    "type_individual", "type_business", "type_shell",
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
    for t in ("individual", "business", "shell"):
        df[f"type_{t}"] = (acc["account_type"] == t).astype(float).reindex(df.index)

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
