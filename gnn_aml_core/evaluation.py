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


def full_report(y_val, p_val, y_test, p_test) -> dict:
    """Threshold picked on validation, metrics reported on test."""
    thr = best_f1_threshold(y_val, p_val)
    return {**rank_metrics(y_test, p_test), **binary_metrics(y_test, p_test, thr)}
