"""Classification metrics used by the trainer, the baselines and the scorecard (numpy + scikit-learn only, no torch)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def rank_metrics(y_true, prob) -> dict:
    """Threshold-free metrics."""
    return {"auc_roc": float(roc_auc_score(y_true, prob)), "pr_auc": float(average_precision_score(y_true, prob))}


def best_f1_threshold(y_true, prob) -> float:
    """Decision threshold that maximises F1 (choose it on VALIDATION data, then apply it to the test set)."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[int(np.argmax(f1[:-1]))])


def binary_metrics(y_true, prob, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(prob) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def precision_targets_table(y_true, prob, targets=(0.5, 0.7, 0.9, 1.0)) -> list[dict]:
    """The mirror of recall_targets_table: for each target precision, the operating point that MAXIMISES
    recall while still meeting that precision -- the real cost of pushing precision toward its own limit,
    the opposite end of the same tradeoff.

    target_precision=1.0 is the mathematical limit here too: achieving it exactly (zero false positives at
    all) generally means flagging only the small handful of cases the model is most confident about, which
    can mean missing the great majority of real positives -- the real, uncomfortable recall cost a "never a
    false alarm" requirement imposes, surfaced directly rather than left implicit.

    Unlike recall (which decreases monotonically as the threshold rises), precision is NOT guaranteed
    monotonic -- removing the single lowest-scoring flagged item can occasionally lower precision if that item
    happened to be a false positive sitting just above a run of true positives. This checks every achievable
    threshold directly rather than assuming monotonicity, and picks the best (highest-recall) one that
    actually meets the target, rather than the first one encountered scanning in one direction.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob)
    candidate_thresholds = np.sort(np.unique(p))

    rows = []
    for target in sorted(targets):
        best = None
        for t in candidate_thresholds:
            m = binary_metrics(y, p, float(t))
            if m["precision"] >= target and (best is None or m["recall"] > best["recall"]):
                best = m
        if best is None:
            # not even the single most-confident prediction reached this precision -- report the highest
            # achievable precision's own operating point instead of silently returning nothing
            best = binary_metrics(y, p, float(p.max()))
        rows.append({"target_precision": target, **best})
    return rows


def recall_targets_table(y_true, prob, targets=(0.90, 0.95, 0.99, 1.0)) -> list[dict]:
    """For each target recall level, the real, achievable operating point: the HIGHEST threshold (so the
    FEWEST false positives) that still reaches at least that recall -- concrete tp/fp/fn/tn counts and the
    resulting precision, not an abstract promise. Answers "how far can this be pushed" directly, rather than
    reporting one single threshold and leaving the actual tradeoff curve invisible.

    target_recall=1.0 is the mathematical limit, not a special case handled differently: achieving it only
    ever means flagging every single row with a score above the absolute minimum positive score (or everyone,
    if even that is not enough) -- the resulting precision at that point is the real, uncomfortable number a
    "catch everything" requirement actually costs, not a hypothetical one.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob)
    candidate_thresholds = np.sort(np.unique(p))[::-1]   # descending: highest score first = fewest flagged first

    rows = []
    for target in sorted(targets):
        best = None
        for t in candidate_thresholds:
            m = binary_metrics(y, p, float(t))
            if m["recall"] >= target:
                best = m   # first match scanning high-to-low threshold = highest threshold = fewest false positives
                break
        if best is None:
            # even flagging every single row did not reach this target (only possible if some positives tie
            # with negatives at the lowest score, or the target itself is unreachable, e.g. > achievable max)
            best = binary_metrics(y, p, float(p.min()) - 1e-9)
        rows.append({"target_recall": target, **best})
    return rows


def full_report(y_val, p_val, y_test, p_test) -> dict:
    """Threshold picked on validation, metrics reported on test."""
    thr = best_f1_threshold(y_val, p_val)
    return {**rank_metrics(y_test, p_test), **binary_metrics(y_test, p_test, thr)}
