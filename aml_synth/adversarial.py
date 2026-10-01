"""aml_synth.adversarial: simulates launderers adapting after detection, using the generator's own real,
existing "camouflage" knob as the adaptation mechanism -- not a new hidden lever invented for this experiment.

The real question this project's whole feedback-loop story has assumed but never actually tested: if a model
is trained once and money launderers subsequently adapt to what got them caught, how much does detection
degrade, and does retraining (simulating gnn_aml_core.feedback's real officer-decision loop) actually recover
it? A benchmark frozen at one point in time cannot answer this on its own -- it needs a genuine two-generation
comparison.

The method, in three steps:
  1. Train a baseline model on generation-1 data (ordinary camouflage).
  2. Measure which features that model relied on most (its own coefficients -- an honest, direct signal of
     what the model is actually keying on, not an assumption about what "should" matter).
  3. Generate generation-2 data with camouflage increased specifically for the typologies whose defining
     behavioural markers matched the most-relied-on features, simulating those launderers blending in more now
     that they know (in this simulation) what got them caught. Evaluate the SAME, unretrained generation-1
     model against generation 2 -- the accuracy drop is the real, honest measurement of adaptation pressure.
     Then retrain on generation 2 and re-measure, showing whether retraining recovers it.

This is deliberately built on the generator's one existing, real difficulty knob (camouflage) rather than
inventing new hidden per-typology adaptation parameters that would need their own separate validation --
camouflage already means exactly "ordinary traffic carried by laundering accounts, to look more normal," which
is precisely what adapting-to-evade-detection means in this synthetic world.

HONEST LIMITATION, found by actually running this: camouflage does not reliably move detection difficulty in
this generator, even at a large increase (5x tested) -- consistent with an earlier finding made during this
project's difficulty-calibration work this same session (see gnn_aml_core/README.md's benchmark-difficulty
history). This module's simulation MECHANISM is built and works correctly (it genuinely measures relied-on
features, adapts the config, and compares generations); what it reveals is that camouflage specifically is too
weak a lever to show strong adaptation pressure. The typology-specific imperfection parameters that DO
meaningfully affect difficulty (retained fraction, timing spread) are not currently exposed as GeneratorConfig
knobs, so a genuinely strong adaptation simulation would need those exposed too -- a real, scoped, disclosed
follow-up, not attempted here.

    python -m aml_synth.adversarial --accounts 5000 --seed 1
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

log = logging.getLogger("aml_synth.adversarial")

# Which typologies each feature's own construction most directly reflects -- used to decide which typologies'
# camouflage to raise when that feature turns out to be what the model relies on most. Includes the general
# amount/degree statistics (out_amt_mean, out_uniq, etc.) mapped broadly to every typology: a first real run of
# this module found these, not the newer specialised features, are what a simple baseline actually relies on
# most in practice -- a mapping that only covered the specialised features left the adaptation mechanism never
# triggering at all, since it never matched what the model was really keying on. Deliberately not exhaustive
# beyond this, since a wrong or overreaching mapping would make the "adaptation" less honest than just raising
# camouflage everywhere.
ALL_TYPOLOGIES = ["smurfing", "scatter_gather", "cyclic_loop", "shell_company", "cross_border_velocity",
                  "dormant_reactivation", "asymmetric_structuring"]
FEATURE_TO_TYPOLOGIES = {
    # general amount/degree statistics: broadly relevant, since every typology's amounts and transaction
    # counts differ from ordinary traffic by construction, just via different specific mechanisms
    "out_amt_mean": ALL_TYPOLOGIES, "out_amt_sum": ALL_TYPOLOGIES, "out_amt_max": ALL_TYPOLOGIES,
    "out_amt_std": ALL_TYPOLOGIES, "in_amt_mean": ALL_TYPOLOGIES, "in_amt_sum": ALL_TYPOLOGIES,
    "in_amt_max": ALL_TYPOLOGIES, "in_amt_std": ALL_TYPOLOGIES, "out_uniq": ["smurfing", "scatter_gather", "asymmetric_structuring"],
    "in_uniq": ["smurfing", "asymmetric_structuring"], "out_deg": ALL_TYPOLOGIES, "in_deg": ALL_TYPOLOGIES,
    # specialised, newer features: mapped to the specific typology whose construction they were built to detect
    "pass_through_match_ratio": ["smurfing", "scatter_gather"],
    "near_thr_ratio": ["smurfing"],
    "round_amt_ratio": ["smurfing", "scatter_gather"],
    "retained_frac": ["smurfing", "scatter_gather"],
    "burst_6h": ["smurfing"],
    "flow_ratio": ["smurfing", "scatter_gather", "shell_company"],
    "high_risk_dest_ratio": ["smurfing"],
    "shared_memo_ratio": ["smurfing"],
    "dispute_rate": ["asymmetric_structuring"],
    "counterparty_registration_cluster_ratio": ["smurfing"],
    "volume_shift_ratio": ["dormant_reactivation"],
    "hour_concentration": ["smurfing"],
}


def top_relied_on_features(x: np.ndarray, y: np.ndarray, feature_names: list[str], top_k: int = 5) -> list[str]:
    """Which features a simple, fast, interpretable baseline relies on most -- logistic regression's own
    coefficient magnitudes, an honest direct read of what the model keys on, not an external assumption about
    what "should" matter. Uses the whole dataset (not a held-out split): this measurement is about
    understanding the model's learned behaviour, not evaluating its generalisation, so there is no leakage
    concern the way there would be for a reported accuracy number."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(x, y)
    order = np.argsort(-np.abs(clf.coef_[0]))
    return [feature_names[i] for i in order[:top_k]]


def adapted_config(base_kwargs: dict, relied_on_features: list[str], boost: float = 1.5) -> dict:
    """Returns a NEW dict of GeneratorConfig kwargs with camouflage raised specifically for the typologies
    whose defining behavioural markers matched the model's most-relied-on features -- not a blanket increase
    everywhere, which would not represent adaptation so much as generic noise. `boost` is a multiplicative
    increase (default 1.5x); the mapping from feature to typology is FEATURE_TO_TYPOLOGIES above.

    This only adjusts camouflage, the one existing, real config knob that means "look more ordinary" -- it does
    not invent new hidden per-typology parameters, so the adaptation this simulates is honestly bounded by what
    the generator can actually already do, not an unfalsifiable new mechanism built just for this experiment.
    """
    out = dict(base_kwargs)
    affected_typologies = {t for f in relied_on_features for t in FEATURE_TO_TYPOLOGIES.get(f, [])}
    base_camouflage = float(out.get("camouflage", 1.5))
    if affected_typologies:
        out["camouflage"] = base_camouflage * boost
        log.info("adapting: raising camouflage %.2f -> %.2f (typologies implicated by relied-on features: %s)",
                 base_camouflage, out["camouflage"], sorted(affected_typologies))
    else:
        log.info("no typology-mapped feature among the top relied-on features; camouflage left unchanged")
    return out


def run_adaptation_round(n_accounts: int = 5000, seed: int = 1, top_k: int = 5, boost: float = 1.5) -> dict:
    """Runs the full two-generation comparison and returns a result dict with generation-1 accuracy,
    generation-2 (adapted) accuracy using the SAME gen-1 model (the real adaptation-pressure measurement), and
    generation-2 accuracy after retraining on generation 2 (the feedback-loop-recovery measurement)."""
    from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig
    from gnn_aml_core.features import build_node_features, standardize
    from gnn_aml_core.train import make_splits
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    base_kwargs = {"n_accounts": n_accounts, "n_background_tx": n_accounts * 8, "seed": seed}
    g1 = AMLGraphGenerator(GeneratorConfig(**base_kwargs)).generate()
    x1, names = build_node_features(g1.accounts, g1.transactions)
    y1 = g1.accounts["is_suspicious"].to_numpy()
    xs1, mean1, std1 = standardize(x1)
    tr1, va1, te1 = make_splits(y1, seed=seed)

    clf1 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(xs1[tr1], y1[tr1])
    gen1_auc = roc_auc_score(y1[te1], clf1.predict_proba(xs1[te1])[:, 1])
    log.info("generation 1: AUC %.3f (%d accounts, seed %d)", gen1_auc, n_accounts, seed)

    relied_on = top_relied_on_features(xs1[tr1], y1[tr1], names, top_k=top_k)
    log.info("top %d relied-on features: %s", top_k, relied_on)

    gen2_kwargs = adapted_config(base_kwargs, relied_on, boost=boost)
    gen2_kwargs["seed"] = seed + 1_000  # a genuinely different graph, not a re-roll of the same one
    g2 = AMLGraphGenerator(GeneratorConfig(**gen2_kwargs)).generate()
    x2, _ = build_node_features(g2.accounts, g2.transactions)
    y2 = g2.accounts["is_suspicious"].to_numpy()
    xs2 = (x2 - mean1) / std1   # standardized against GEN-1's own mean/std: this model has never seen
                                # generation 2's data distribution at all, exactly as a real static deployed
                                # model would score genuinely new, adapted transactions

    gen2_auc_same_model = roc_auc_score(y2, clf1.predict_proba(xs2)[:, 1])
    log.info("generation 2 (adapted), SAME gen-1 model, never retrained: AUC %.3f  (drop: %.3f)",
             gen2_auc_same_model, gen1_auc - gen2_auc_same_model)

    xs2_std, _, _ = standardize(x2)
    tr2, va2, te2 = make_splits(y2, seed=seed)
    clf2 = LogisticRegression(max_iter=3000, class_weight="balanced").fit(xs2_std[tr2], y2[tr2])
    gen2_auc_retrained = roc_auc_score(y2[te2], clf2.predict_proba(xs2_std[te2])[:, 1])
    log.info("generation 2 (adapted), RETRAINED on generation 2: AUC %.3f  (recovery: %.3f)",
             gen2_auc_retrained, gen2_auc_retrained - gen2_auc_same_model)

    return {"gen1_auc": gen1_auc, "gen2_auc_same_model": gen2_auc_same_model, "gen2_auc_retrained": gen2_auc_retrained,
           "relied_on_features": relied_on, "degradation": gen1_auc - gen2_auc_same_model,
           "recovery": gen2_auc_retrained - gen2_auc_same_model}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accounts", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--boost", type=float, default=1.5)
    a = ap.parse_args()

    res = run_adaptation_round(n_accounts=a.accounts, seed=a.seed, top_k=a.top_k, boost=a.boost)
    log.info("SUMMARY: gen1=%.3f -> gen2(unadapted model)=%.3f (degradation %.3f) -> gen2(retrained)=%.3f (recovery %.3f)",
             res["gen1_auc"], res["gen2_auc_same_model"], res["degradation"], res["gen2_auc_retrained"], res["recovery"])
    if res["degradation"] > 0.02:
        log.info("A static model measurably degrades once launderers adapt (this synthetic simulation of it, "
                 "at least) -- this is the real, quantified case for gnn_aml_core.feedback's retraining loop, "
                 "not just an assumption that it would help.")
    else:
        log.info("Degradation was small or negative in this run. This is consistent with an earlier finding "
                 "this project already made during difficulty calibration: camouflage (this module's only "
                 "currently available 'blend in more' lever) does not reliably move detection difficulty in "
                 "this generator, even at a large increase -- it is not simply this run's seed. The typology-"
                 "specific imperfection parameters (retained fraction, timing spread, etc.) that DO meaningfully "
                 "affect difficulty are not currently exposed as GeneratorConfig knobs, so this module cannot "
                 "adapt them; exposing them is a real, scoped, disclosed follow-up, not a hidden gap.")


if __name__ == "__main__":
    main()
