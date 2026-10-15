"""Baseline models the GNN has to beat (brief 2.3): XGBoost, Logistic Regression, Isolation Forest,
plus a strong extra baseline: boosting on each account's own features AND its neighbourhood averages.

Only numpy / scikit-learn (and xgboost if installed). Everything is evaluated exactly like the GNN:
threshold chosen on the validation split, metrics reported on the test split.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

from gnn_aml_core.evaluation import full_report


def neighbour_aggregates(x: np.ndarray, src: np.ndarray, dst: np.ndarray, hops: int = 2) -> np.ndarray:
    """[own features | mean of 1-hop neighbours | mean of 2-hop neighbours], ignoring edge direction."""
    n = x.shape[0]
    m = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    m = m + m.T
    deg = np.asarray(m.sum(1)).ravel()
    p = sp.diags(1.0 / np.maximum(deg, 1)) @ m
    parts, cur = [x], x
    for _ in range(hops):
        cur = p @ cur
        parts.append(np.asarray(cur))
    return np.hstack(parts).astype(np.float32)


def _boosting(y_train, seed):
    """XGBoost if installed (as in the brief); otherwise scikit-learn's histogram gradient boosting."""
    try:
        from xgboost import XGBClassifier
        pos = max(int(y_train.sum()), 1)
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=(len(y_train) - pos) / pos, eval_metric="logloss", tree_method="hist",
                             n_jobs=2, random_state=seed), "XGBoost"
    except ImportError:
        return HistGradientBoostingClassifier(random_state=seed), "HistGradientBoosting (XGBoost not installed)"


def evaluate_baselines(x, y, idx_train, idx_val, idx_test, edges=None, seed: int = 0) -> dict:
    """x: standardised node features [N, F]; y: labels [N]; idx_*: split indices; edges: optional (src, dst) for the graph-aware baseline."""
    out, y = {}, np.asarray(y)

    def run(name, model, xs, scorer="proba"):
        model.fit(xs[idx_train], y[idx_train]) if scorer == "proba" else model.fit(xs[idx_train])
        score = (lambda ix: model.predict_proba(xs[ix])[:, 1]) if scorer == "proba" else (lambda ix: -model.score_samples(xs[ix]))
        out[name] = full_report(y[idx_val], score(idx_val), y[idx_test], score(idx_test))

    run("Logistic Regression", LogisticRegression(max_iter=3000, class_weight="balanced"), x)
    gb, gb_name = _boosting(y[idx_train], seed)
    run(gb_name, gb, x)
    run("Isolation Forest (unsupervised score)", IsolationForest(random_state=seed, contamination=0.05), x, scorer="anomaly")
    if edges is not None:
        xg = neighbour_aggregates(x, edges[0], edges[1])
        gb2, nm = _boosting(y[idx_train], seed)
        run(f"{nm} + neighbour averages (strong, uses the graph)", gb2, xg)
    return out
