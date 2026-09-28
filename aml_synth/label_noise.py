"""aml_synth.label_noise: simulates realistic, messy investigator labeling on top of the generator's perfect
ground truth.

Every other part of this project trains and evaluates against `is_suspicious`, which is perfect by
construction (we injected the pattern, so we know exactly which accounts are involved). Real SAR investigation
labels are not like this: investigators miss genuinely suspicious accounts (false negatives -- the pattern was
there, no one caught it), and occasionally file a SAR on an account that turns out to be innocent (false
positives -- a wrongful report). A model that only ever sees perfect labels tells you nothing about how it
would perform trained on the labels a real bank actually has.

This module never overwrites the ground truth. It adds a *second* column, `is_suspicious_noisy`, so you can
train on the noisy column and evaluate against the true one -- the real, useful research question ("how much
does realistic label noise hurt detection performance") that a single corrupted column could never answer,
since you would have thrown away the ability to check against truth at all.

    from aml_synth.label_noise import inject_label_noise
    accounts = inject_label_noise(graph.accounts, false_negative_rate=0.15, false_positive_rate=0.02, seed=1)
    # train on accounts["is_suspicious_noisy"], evaluate against accounts["is_suspicious"]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inject_label_noise(accounts: pd.DataFrame, false_negative_rate: float = 0.15,
                       false_positive_rate: float = 0.02, seed: int | None = None) -> pd.DataFrame:
    """Returns a COPY of `accounts` with a new `is_suspicious_noisy` column layered on top of the true
    `is_suspicious` column, simulating realistic investigator error:

    false_negative_rate: share of genuinely suspicious accounts whose noisy label flips to 0 (a real pattern
        an investigator never caught -- FinCEN's own SAR effectiveness studies suggest this is common;
        0.15 is a starting estimate, not a citation, and should be treated as a knob to experiment with).
    false_positive_rate: share of genuinely benign accounts whose noisy label flips to 1 (a wrongful SAR filed
        on an innocent account -- rarer than a missed case, hence the much lower default rate).

    Raises ValueError for a rate outside [0, 1] -- a silent clip would hide a real mistake (e.g. passing a
    percentage like 15 instead of a fraction like 0.15).
    """
    for name, rate in (("false_negative_rate", false_negative_rate), ("false_positive_rate", false_positive_rate)):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{name}={rate} must be between 0.0 and 1.0 (did you pass a percentage instead of a fraction?)")

    rng = np.random.default_rng(seed)
    out = accounts.copy()
    true_label = out["is_suspicious"].to_numpy()
    noisy = true_label.copy()

    suspicious_idx = np.where(true_label == 1)[0]
    benign_idx = np.where(true_label == 0)[0]

    flip_to_negative = rng.random(len(suspicious_idx)) < false_negative_rate
    noisy[suspicious_idx[flip_to_negative]] = 0

    flip_to_positive = rng.random(len(benign_idx)) < false_positive_rate
    noisy[benign_idx[flip_to_positive]] = 1

    out["is_suspicious_noisy"] = noisy
    return out


def label_noise_report(accounts: pd.DataFrame) -> dict:
    """Summary statistics comparing `is_suspicious` (true) against `is_suspicious_noisy`, for a sanity check
    after calling inject_label_noise, or for reporting alongside a benchmark table that used noisy labels."""
    if "is_suspicious_noisy" not in accounts.columns:
        raise KeyError("accounts has no 'is_suspicious_noisy' column -- call inject_label_noise first")
    true_label = accounts["is_suspicious"].to_numpy()
    noisy = accounts["is_suspicious_noisy"].to_numpy()
    n_suspicious = int((true_label == 1).sum())
    n_benign = int((true_label == 0).sum())
    n_false_neg = int(((true_label == 1) & (noisy == 0)).sum())
    n_false_pos = int(((true_label == 0) & (noisy == 1)).sum())
    return {
        "n_accounts": len(accounts), "n_truly_suspicious": n_suspicious, "n_truly_benign": n_benign,
        "n_false_negatives": n_false_neg, "n_false_positives": n_false_pos,
        "false_negative_rate_realised": n_false_neg / n_suspicious if n_suspicious else 0.0,
        "false_positive_rate_realised": n_false_pos / n_benign if n_benign else 0.0,
        "noisy_positive_count": int((noisy == 1).sum()),
    }
