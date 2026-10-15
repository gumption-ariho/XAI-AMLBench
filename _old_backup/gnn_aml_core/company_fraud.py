"""gnn_aml_core.company_fraud: loads an externally-sourced company/transaction fraud benchmark and evaluates our
baseline classifiers against it.

IMPORTANT, read before using this module -- same honest scoping as gnn_aml_core.synthaml: this dataset has NO
graph structure at all. There is no company-to-company edge, no transaction-to-transaction edge, nothing for a
graph neural network to operate on. `gnn_aml_core.train.fit_gnn` (GATv2/RGCN) CANNOT be run on this data, and
this module does not attempt to force it to -- it evaluates `gnn_aml_core.baselines.evaluate_baselines`'s
non-graph models only.

I was not able to identify this dataset's original source or a citable paper for it (the person who found it
could not either, and a search for its distinctive file names did not turn up a match). It is used here purely
as a tabular baseline sanity check, not represented as validated against a known, peer-reviewed benchmark the
way gnn_aml_core.elliptic and gnn_aml_core.synthaml are.

Data: five files, schema confirmed directly against the real files before writing this loader (not assumed):

  companies_{train,test}.csv      one row per company, an unnamed index column (the company id) + 6 anonymised
                                   numeric columns (named "0".."5" as plain strings, not descriptive names)
  transactions_{train,test}.csv   one row per transaction: transactionId + 43 anonymised numeric columns
                                   (named "0".."42")
  event_order_{train,test}.csv    transactionId -> eventAt (a sequence position, NOT a real timestamp)
  time_series_ids_{train,test}.csv  transactionId -> a "company_X_window_Y" key. THIS IS A MANY-TO-MANY JOIN,
                                   not one window per transaction: on the real training data, 2,343,143 mappings
                                   exist for only 1,503,611 transactions (confirmed by direct inspection), so a
                                   transaction can legitimately belong to more than one window (consistent with
                                   overlapping/sliding windows) -- this loader does not assume a 1:1 mapping.
  fraud_labels_{train,test}.csv   an unnamed index column (the "company_X_window_Y" key) + isFraudUser
                                   (True/False). This is the actual prediction target, and it is NOT per
                                   transaction -- it is per (company, time window). The real task this dataset
                                   asks is "did this company's activity during this window look fraudulent",
                                   not "is this one transaction fraudulent".

Train/test asymmetry, confirmed directly: train has ~7 windows per company on average (5,428 companies, 38,870
labelled windows); test has exactly one window per company (2,305 companies, 2,305 labelled windows) -- test is
NOT a smaller sample of the same distribution as train in that respect, and this loader does not pretend it is.

Join integrity, confirmed directly: every company referenced in fraud_labels_train.csv has a matching row in
companies_train.csv (zero orphans) -- this loader still does not silently assume that holds for data it has not
seen (a differently-exported copy, a future version), and handles a missing join gracefully rather than
crashing or silently dropping rows without saying so.

    python -m gnn_aml_core.company_fraud --data data/company_fraud
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("gnn_aml_core.company_fraud")

# Summary statistics computed per (company, window) over that window's transactions. Deliberately the same
# small, standard, interpretable set gnn_aml_core.synthaml already uses for its own per-alert aggregation, not
# a larger set invented just for this dataset.
AGG_STATS = ("mean", "std", "min", "max", "count")


def load_split(data_dir: Path, split: str) -> dict[str, pd.DataFrame]:
    """Loads one split ("train" or "test") of all 5 files. companies and fraud_labels both have an unnamed
    first column that is really their key (the company id, and the "company_X_window_Y" key respectively) --
    index_col=0 makes that explicit rather than leaving it as a confusing "Unnamed: 0" column."""
    d = data_dir
    return {
        "companies": pd.read_csv(d / f"companies_{split}.csv", index_col=0),
        "transactions": pd.read_csv(d / f"transactions_{split}.csv"),
        "event_order": pd.read_csv(d / f"event_order_{split}.csv"),
        "time_series_ids": pd.read_csv(d / f"time_series_ids_{split}.csv"),
        "fraud_labels": pd.read_csv(d / f"fraud_labels_{split}.csv", index_col=0),
    }


def build_window_features(split: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X float32 [N, F], y [N], feature_names) for every LABELLED (company, window) in this split.
    Row order matches split["fraud_labels"].index order.

    The real join, in order: transactions -> time_series_ids (many-to-many: one transaction can belong to
    several windows, so transactions are deliberately not deduplicated against this join) -> group by window key
    -> aggregate -> join the window's company's own static features (by parsing "company_X_window_Y" back to
    "company_X", the only way to recover the company id, since it is not carried as its own column anywhere) ->
    align to fraud_labels' index, which is the authoritative set of labelled windows.
    """
    tx = split["transactions"]
    tsid = split["time_series_ids"]
    companies = split["companies"]
    labels = split["fraud_labels"]

    tx_cols = [c for c in tx.columns if c != "transactionId"]
    joined = tsid.merge(tx, on="transactionId", how="left")
    # a transactionId in time_series_ids with no matching row in transactions.csv would silently produce NaN
    # feature columns for every stat below (mean/std/etc. of all-NaN) rather than crash -- that is deliberate:
    # fillna(0.0) at the end treats "we have no transaction data for this window" the same honest way
    # gnn_aml_core.features already treats "no transactions for this account" elsewhere in this project.

    grouped = joined.groupby("time_series_ids")[tx_cols].agg(list(AGG_STATS))
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    # std of a single-transaction window is NaN by construction (pandas), not a real missing value -- the same
    # real bug gnn_aml_core.synthaml already hit and fixed once in this project; filled to 0.0 here from the
    # start rather than waiting to rediscover it.
    grouped = grouped.fillna(0.0)

    company_id = pd.Series(labels.index, index=labels.index).str.replace(r"_window_\d+$", "", regex=True)
    company_feats = companies.reindex(company_id).reset_index(drop=True)
    company_feats.columns = [f"company_{c}" for c in company_feats.columns]
    company_feats.index = labels.index
    # a window whose company id is not in companies.csv at all (confirmed zero such cases on the real training
    # data, but this loader does not assume that holds for data it has not seen) gets 0.0 company features
    # rather than crashing or silently dropping the row -- consistent with the rest of this handling.
    company_feats = company_feats.fillna(0.0)

    window_feats = grouped.reindex(labels.index).fillna(0.0)
    full = pd.concat([window_feats, company_feats], axis=1)

    x = full.to_numpy(dtype=np.float32)
    y = labels["isFraudUser"].astype(bool).astype(int).to_numpy()
    return x, y, list(full.columns)


def run(data_dir: Path, seed: int = 42) -> None:
    from gnn_aml_core.baselines import evaluate_baselines
    from gnn_aml_core.features import standardize
    from gnn_aml_core.train import format_table

    train = load_split(data_dir, "train")
    test = load_split(data_dir, "test")

    x_train, y_train, names = build_window_features(train)
    x_test, y_test, _ = build_window_features(test)
    log.info("train: %d labelled windows, %d features (%d fraudulent, %.1f%%)",
             len(y_train), len(names), int(y_train.sum()), 100 * y_train.mean())
    log.info("test:  %d labelled windows (%d fraudulent, %.1f%%)",
             len(y_test), int(y_test.sum()), 100 * y_test.mean())

    # evaluate_baselines takes ONE combined matrix plus index arrays into it, not separate train/test matrices
    # (gnn_aml_core.synthaml's own real, proven usage) -- train and test here come from the source's own files
    # as genuinely separate splits (not drawn from the same pool), so they are concatenated and val is carved
    # out of train only, exactly as gnn_aml_core.synthaml does for its own quarter-based split, never touching
    # the real test rows.
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    idx_train_full = np.arange(len(y_train))
    idx_test = np.arange(len(y_train), len(y_train) + len(y_test))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx_train_full)
    cut = max(1, int(0.8 * len(perm)))
    idx_train, idx_val = perm[:cut], perm[cut:]
    log.info("split: train %d / val %d / test %d", len(idx_train), len(idx_val), len(idx_test))

    xs, _, _ = standardize(x)
    table = evaluate_baselines(xs, y, idx_train, idx_val, idx_test, edges=None, seed=seed)
    log.info("REAL-DATA RESULTS (tabular only -- no graph model applies here; see this module's docstring)\n%s",
             format_table(table))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                   help="directory containing companies_{train,test}.csv and the other 4 file pairs")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(a.data, seed=a.seed)


if __name__ == "__main__":
    main()
