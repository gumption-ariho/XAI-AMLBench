"""gnn_aml_core.calibration: post-hoc probability calibration.

A GNN trained with heavily class-weighted loss (needed here because only a small share of accounts are
suspicious) tends to push its raw sigmoid outputs to the extremes: a flagged account often scores 0.99+
regardless of how confident the model really is, because the loss rewards decisiveness far more than it
penalises overconfidence. This does not hurt ranking metrics at all -- AUC-ROC, PR-AUC, and precision/recall at
a fixed threshold depend only on the ORDER of scores, not their absolute values -- but it makes the score
meaningless as a probability, and uninformative on a UI risk gauge where every flagged account reads "99.9%".

Isotonic regression fixes this: it fits a monotonic (rank-preserving) step function mapping raw scores to
calibrated probabilities that better match the empirical frequency of the positive class, without changing
which account is considered more or less risky than another. Fit it on the VALIDATION split only -- never on
train (would overfit) or test (would leak test information into what should be an honest final evaluation).
"""
from __future__ import annotations

import numpy as np


class Calibrator:
    """A monotonic raw-score -> calibrated-probability mapping, stored as plain breakpoint arrays so it survives
    torch.save/torch.load (or json) without needing scikit-learn at load time -- only numpy's np.interp."""

    def __init__(self, x_thresholds, y_thresholds):
        self.x_thresholds = np.asarray(x_thresholds, dtype=np.float64)
        self.y_thresholds = np.asarray(y_thresholds, dtype=np.float64)

    def __call__(self, raw_scores):
        """Map raw score(s) to calibrated probabilities. Accepts a scalar or an array; returns the same shape."""
        scalar_input = np.isscalar(raw_scores)
        out = np.interp(np.asarray(raw_scores, dtype=np.float64), self.x_thresholds, self.y_thresholds)
        return float(out) if scalar_input else out

    def to_dict(self) -> dict:
        return {"x_thresholds": self.x_thresholds.tolist(), "y_thresholds": self.y_thresholds.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(d["x_thresholds"], d["y_thresholds"])

    @classmethod
    def identity(cls) -> "Calibrator":
        """A no-op calibrator (returns its input unchanged). Used as a safe fallback when no calibration data
        was available at training time, so callers never need a separate code path for "uncalibrated"."""
        return cls([0.0, 1.0], [0.0, 1.0])


def fit_calibrator(raw_scores, y_true) -> Calibrator:
    """Fit isotonic regression on (raw_scores, y_true) from a VALIDATION split and return a Calibrator."""
    from sklearn.isotonic import IsotonicRegression

    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_scores, y_true)
    return Calibrator(iso.X_thresholds_, iso.y_thresholds_)


def expected_calibration_error(probs, y_true, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted probability and empirical frequency, over `n_bins` equal-width bins
    of predicted probability, weighted by how many points fall in each bin. 0.0 is perfect calibration; this is
    a diagnostic only (nothing is fit to minimise it directly)."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    if n == 0:
        return 0.0
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(probs[mask].mean() - y_true[mask].mean())
    return float(ece)


def brier_score(probs, y_true) -> float:
    """Mean squared error between predicted probability and the (0/1) outcome. Lower is better; 0.0 is perfect."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((probs - y_true) ** 2))
