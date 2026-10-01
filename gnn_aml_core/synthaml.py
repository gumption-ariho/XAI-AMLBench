"""gnn_aml_core.synthaml: loads SynthAML (Jensen et al., 2023, Nature Scientific Data), a real, bank-derived
AML benchmark, and evaluates our baseline classifiers against it.

IMPORTANT, read before using this module: unlike Elliptic, SynthAML has NO graph structure at all. Each row is
one client's transaction HISTORY leading up to an AML alert (up to 365 days of transactions for that one
client) -- there is no src/dst, no account-to-account edges, nothing for a graph neural network to operate on.
The paper's own methodology confirms this: it engineers 56 hand-crafted summary statistics per alert and feeds
them to classical tabular models (decision tree, random forest, logistic regression, SVM, MLP, LightGBM), never
a graph model. `gnn_aml_core.train.fit_gnn` (GATv2/RGCN) CANNOT be run on this data, and this module does not
attempt to force it to -- it evaluates `gnn_aml_core.baselines.evaluate_baselines`'s non-graph models only.

What this genuinely adds: SynthAML is built from a real bank's (Spar Nord, Denmark) real transaction data and
real AML alert outcomes -- the right domain (bank wire/card/cash activity, not Bitcoin like Elliptic) with a
peer-reviewed claim that performance on it transfers to the real world. Running our own baseline
implementations against it is a genuine, real-data-derived sanity check on those baselines specifically,
directly comparable to the paper's own reported numbers (their Table 3).

Data: two files, "synthetic_alerts.csv" and "synthetic_transactions.csv", from
https://doi.org/10.6084/m9.figshare.c.6504421.v1 (open access, CC-BY 4.0).

  synthetic_alerts.csv: one row per alert -- an alert id, the date raised, and the outcome (reported/dismissed).
  synthetic_transactions.csv: one row per transaction -- the alert id it belongs to, a timestamp, entry
    (credit/debit), type (card/cash/international/wire), and a standardized log-transaction-size.

Per the paper: dates are only accurate to the quarter, so any train/test split must fall on a quarter boundary
(Jan 1, Apr 1, Jul 1, Oct 1) -- this module enforces that rather than accepting an arbitrary cutoff date.

    python -m gnn_aml_core.synthaml --data data/synthaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("gnn_aml_core.synthaml")

RAW_FILES = ("synthetic_alerts.csv", "synthetic_transactions.csv")
TX_TYPES = ("card", "cash", "international", "wire")
ENTRIES = ("credit", "debit")
QUARTER_STARTS = {1: "-01-01", 4: "-04-01", 7: "-07-01", 10: "-10-01"}


def _require_files(data_dir: Path) -> None:
    missing = [f for f in RAW_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} in {data_dir}. Download SynthAML's two CSV files from "
            "https://doi.org/10.6084/m9.figshare.c.6504421.v1 (open access) and place them in this folder."
        )


def _check_quarter_boundary(date_str: str) -> None:
    suffix = date_str[4:]
    if suffix not in QUARTER_STARTS.values():
        raise ValueError(
            f"'{date_str}' is not a quarter boundary (YYYY-01-01, YYYY-04-01, YYYY-07-01 or YYYY-10-01). "
            "SynthAML's alert dates are only accurate to the quarter (see the paper's Usage Notes); a split "
            "on any other date would not mean what it appears to."
        )


def build_alert_features(alerts: pd.DataFrame, tx: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Replicates the paper's own feature engineering exactly (their "Machine learning experiments" subsection):
    for each alert, the (min, mean, median, max, std, count, sum) of transaction size, computed separately for
    each of the 4 transaction types x 2 entry types = 8 groups, giving 7 x 8 = 56 features per alert. A group
    with zero transactions gets count=0 and every other statistic set to -3 (the paper's own convention, chosen
    because -3 sits below the minimum standardized transaction size in their data).

    Returns (X, feature_names, y, dates) where y is the alert outcome (1 = reported) and dates are each alert's
    raised-date, needed for a quarter-respecting train/test split.
    """
    stats = ("min", "mean", "median", "max", "std", "count", "sum")
    feature_names = [f"{t}_{e}_{s}" for t in TX_TYPES for e in ENTRIES for s in stats]

    tx = tx.merge(alerts[["alert_id"]], on="alert_id", how="inner")
    grouped = tx.groupby(["alert_id", "type", "entry"])["amount"].agg(list(stats))

    n = len(alerts)
    x = np.full((n, len(feature_names)), -3.0, dtype=np.float64)
    alert_pos = {aid: i for i, aid in enumerate(alerts["alert_id"])}

    for (aid, t, e), row in grouped.iterrows():
        if aid not in alert_pos or t not in TX_TYPES or e not in ENTRIES:
            continue
        i = alert_pos[aid]
        base = feature_names.index(f"{t}_{e}_min")
        for k, stat in enumerate(stats):
            val = row[stat]
            if stat == "std" and pd.isna(val):
                val = 0.0   # pandas' sample std is undefined (NaN) for a group of exactly one transaction --
                            # zero variance is the correct, meaningful value here, not a missing-data sentinel
            x[i, base + k] = val
    # count defaults to -3 like everything else above; the paper specifies count=0 for an empty group
    for t in TX_TYPES:
        for e in ENTRIES:
            j = feature_names.index(f"{t}_{e}_count")
            x[:, j] = np.where(x[:, j] == -3.0, 0.0, x[:, j])

    y = alerts["outcome"].to_numpy(dtype=np.float64)
    dates = pd.to_datetime(alerts["date"]).to_numpy()
    return x, feature_names, y, dates


def load_synthaml(data_dir: str | Path) -> dict:
    """Reads the two raw SynthAML files and returns alert-level features ready for
    `gnn_aml_core.baselines.evaluate_baselines` (NOT for fit_gnn -- see this module's docstring for why)."""
    data_dir = Path(data_dir)
    _require_files(data_dir)

    alerts = pd.read_csv(data_dir / "synthetic_alerts.csv")
    tx = pd.read_csv(data_dir / "synthetic_transactions.csv")
    alerts.columns = [c.strip().lower() for c in alerts.columns]
    tx.columns = [c.strip().lower() for c in tx.columns]
    rename_alerts = {"id": "alert_id", "raised": "date", "reported": "outcome"}
    rename_tx = {"id": "alert_id", "size": "amount", "amount_dkk": "amount", "timestamp": "timestamp"}
    alerts = alerts.rename(columns={k: v for k, v in rename_alerts.items() if k in alerts.columns})
    tx = tx.rename(columns={k: v for k, v in rename_tx.items() if k in tx.columns})

    for col, frame, name in (("alert_id", alerts, "synthetic_alerts.csv"), ("date", alerts, "synthetic_alerts.csv"),
                             ("outcome", alerts, "synthetic_alerts.csv"), ("alert_id", tx, "synthetic_transactions.csv"),
                             ("type", tx, "synthetic_transactions.csv"), ("entry", tx, "synthetic_transactions.csv"),
                             ("amount", tx, "synthetic_transactions.csv")):
        if col not in frame.columns:
            raise KeyError(f"expected column '{col}' in {name}, found {list(frame.columns)}. "
                           "The real file's column names may differ slightly from what this loader assumes -- "
                           "check the actual header and adjust the rename maps above if needed.")

    alerts["outcome"] = alerts["outcome"].astype(float)
    tx["type"] = tx["type"].astype(str).str.strip().str.lower()
    tx["entry"] = tx["entry"].astype(str).str.strip().str.lower()

    x, feature_names, y, dates = build_alert_features(alerts, tx)
    log.info("SynthAML: %d alerts (%d reported, %.1f%%), %d transactions, %d features",
             len(alerts), int(y.sum()), 100 * y.mean(), len(tx), x.shape[1])
    return {"x": x, "feature_names": feature_names, "y": y, "dates": dates, "n_alerts": len(alerts), "n_tx": len(tx)}


def quarter_split(arrays: dict, train_end: str, test_start: str, test_end: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The paper's own split: train on alerts raised in [start of data, train_end), test on
    [test_start, test_end). All three dates must fall on a quarter boundary. Returns (train_idx, val_idx,
    test_idx) -- val is carved out of the training period (last available quarter of it) since the paper itself
    only reports a train/test split, not a three-way one; a held-out validation slice is still needed here for
    threshold selection, consistent with how every other model in this project chooses its threshold."""
    for d in (train_end, test_start, test_end):
        _check_quarter_boundary(d)
    dates = pd.to_datetime(arrays["dates"])
    train_end_ts, test_start_ts, test_end_ts = pd.Timestamp(train_end), pd.Timestamp(test_start), pd.Timestamp(test_end)
    train_mask = dates < train_end_ts
    test_mask = (dates >= test_start_ts) & (dates < test_end_ts)
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if len(train_idx) == 0:
        raise ValueError("no alerts fall before train_end -- check the date range in your data")
    # carve the val split from the last quarter of the training period
    val_start = train_end_ts - pd.DateOffset(months=3)
    val_mask = train_mask & (dates >= val_start)
    tr_final = np.where(train_mask & ~val_mask)[0]
    va_final = np.where(val_mask)[0]
    if len(tr_final) == 0 or len(va_final) == 0:
        # too little data to carve a separate quarter for validation -- fall back to a random 80/20 split of
        # the training period instead of failing outright
        rng = np.random.default_rng(0)
        perm = rng.permutation(train_idx)
        cut = max(1, int(0.8 * len(perm)))
        tr_final, va_final = perm[:cut], perm[cut:]
    return tr_final, va_final, test_idx


def main() -> None:
    from gnn_aml_core.baselines import evaluate_baselines
    from gnn_aml_core.train import format_table

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-end", default="2021-01-01", help="quarter boundary; train on alerts before this date")
    ap.add_argument("--test-start", default="2021-01-01")
    ap.add_argument("--test-end", default="2022-01-01")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    arrays = load_synthaml(a.data)
    tr, va, te = quarter_split(arrays, a.train_end, a.test_start, a.test_end)
    log.info("split: train %d / val %d / test %d", len(tr), len(va), len(te))

    from gnn_aml_core.features import standardize
    xs, _, _ = standardize(arrays["x"])
    table = evaluate_baselines(xs, arrays["y"], tr, va, te, edges=None, seed=a.seed)
    log.info("REAL-DATA RESULTS on SynthAML (a real Danish bank's alert data, tabular -- no graph model applies "
             "here; see this module's docstring)\n%s", format_table(table))
    log.info("Compare against the paper's own Table 3 (mean ROC AUC across 10 seeds, ordered by synthetic-test "
             "performance) for the real-world-transferability claim this dataset is known for -- this module "
             "does not reproduce those exact figures here, since they were not directly available when this "
             "loader was built; see https://doi.org/10.1038/s41597-023-02569-2 for the real reported numbers.")


if __name__ == "__main__":
    main()
