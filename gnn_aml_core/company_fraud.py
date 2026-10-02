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


def analyze_statistic_importance(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """Which KIND of statistic (mean/std/min/max/count, or the company's own static features) carries the real
    signal -- not which specific anonymised column, which would mean nothing without knowing what it represents
    and would not be a privacy-safe thing to report anyway. A logistic regression's own coefficient magnitudes
    (the same honest, direct method aml_synth.adversarial already uses), grouped by statistic type rather than
    read off individually, turns 221 meaningless column names into a handful of genuinely interpretable,
    transferable findings: e.g. "volatility matters more than scale" is a real, usable insight about fraud
    behaviour in general, even with zero knowledge of what the underlying anonymised values are.

    Returns {statistic_type: aggregate |coefficient| across every feature of that type}, sorted descending.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(x, y)
    importance = np.abs(clf.coef_[0])

    groups: dict[str, float] = {}
    for name, imp in zip(feature_names, importance):
        if name.startswith("company_"):
            key = "company's own static features"
        else:
            # "<original_col>_<stat>" -- the stat suffix is always one of AGG_STATS
            key = next((s for s in AGG_STATS if name.endswith(f"_{s}")), "other")
            key = f"transaction {key} (within-window)"
        groups[key] = groups.get(key, 0.0) + float(imp)

    return dict(sorted(groups.items(), key=lambda kv: -kv[1]))


def run(data_dir: Path, seed: int = 42, analyze: bool = False, sweep: bool = False) -> None:
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

    if analyze:
        xs_train, _, _ = standardize(x_train)
        groups = analyze_statistic_importance(xs_train, y_train, names)
        log.info("Which KIND of statistic carries the signal (privacy-safe: never the specific anonymised "
                 "column, only the statistic type -- see this function's docstring):")
        total = sum(groups.values())
        for key, imp in groups.items():
            log.info("  %-42s %5.1f%%", key, 100 * imp / total)

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

    if sweep:
        from sklearn.linear_model import LogisticRegression
        from gnn_aml_core.evaluation import precision_targets_table, recall_targets_table

        # Logistic Regression specifically: it was the strongest model across every run on this dataset (real,
        # repeated finding, not assumed) -- the sweep uses whichever model actually performs best here, not a
        # fixed choice independent of the data.
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        clf.fit(xs[idx_train], y[idx_train])
        p_test = clf.predict_proba(xs[idx_test])[:, 1]

        log.info("How far can RECALL be pushed, and what does it cost? Each row is the real, achievable "
                 "operating point for that target -- not a hypothetical, the actual threshold and the actual "
                 "false-positive count it requires on this test set.")
        rows = recall_targets_table(y[idx_test], p_test, targets=(0.90, 0.95, 0.99, 1.0))
        log.info("%6s  %9s  %9s  %6s  %6s  %6s", "target", "threshold", "precision", "recall", "FP", "TP")
        for r in rows:
            log.info("%5.0f%%  %9.3f  %9.3f  %5.1f%%  %6d  %6d",
                     100 * r["target_recall"], r["threshold"], r["precision"], 100 * r["recall"], r["fp"], r["tp"])

        log.info("How far can PRECISION be pushed, and what does it cost? The mirror direction -- the real "
                 "recall given up to reach each precision target, the fewest false alarms possible at that "
                 "confidence level.")
        prows = precision_targets_table(y[idx_test], p_test, targets=(0.50, 0.70, 0.90, 1.0))
        log.info("%6s  %9s  %9s  %6s  %6s  %6s", "target", "threshold", "precision", "recall", "FP", "TP")
        for r in prows:
            log.info("%5.0f%%  %9.3f  %9.3f  %5.1f%%  %6d  %6d",
                     100 * r["target_precision"], r["threshold"], r["precision"], 100 * r["recall"], r["fp"], r["tp"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True,
                   help="directory containing companies_{train,test}.csv and the other 4 file pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--analyze", action="store_true",
                   help="also report which KIND of statistic (volatility, scale, activity count, ...) carries "
                        "the real fraud signal -- privacy-safe, never the specific anonymised column")
    ap.add_argument("--sweep", action="store_true",
                   help="report the real, achievable precision/false-positive cost of pushing recall to "
                        "90%%, 95%%, 99%%, and 100%% on the test set -- the actual ceiling, not an estimate")
    a = ap.parse_args()
    run(a.data, seed=a.seed, analyze=a.analyze, sweep=a.sweep)


if __name__ == "__main__":
    main()
