"""Tests for the pure, torch-free logic in gnn_aml_core.benchmark: aggregation, paired comparison, target
counting and the markdown table. `main()` itself runs the full generate -> train -> evaluate pipeline and needs
PyTorch; it is exercised end to end by running `python -m gnn_aml_core.benchmark` manually, not by this suite."""
import numpy as np
import pytest

from gnn_aml_core.benchmark import TARGETS, aggregate, markdown_table, meets_targets, paired, strongest_baseline


def row(auc, precision, recall, f1, fpr, pr_auc=0.9):
    return {"auc_roc": auc, "pr_auc": pr_auc, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


PER_SEED = [
    {"GNN (GATV2)": row(0.988, 0.938, 0.824, 0.877, 0.005),
     "LR": row(0.867, 0.598, 0.538, 0.566, 0.034),
     "Boost + neighbour averages": row(0.973, 0.756, 0.747, 0.751, 0.023, pr_auc=0.857)},
    {"GNN (GATV2)": row(0.981, 0.910, 0.801, 0.852, 0.007),   # recall 0.801 misses the 0.82 target
     "LR": row(0.850, 0.580, 0.500, 0.540, 0.036),
     "Boost + neighbour averages": row(0.965, 0.740, 0.720, 0.730, 0.025, pr_auc=0.84)},
    {"GNN (GATV2)": row(0.990, 0.945, 0.840, 0.889, 0.004),
     "LR": row(0.870, 0.610, 0.550, 0.580, 0.033),
     "Boost + neighbour averages": row(0.975, 0.770, 0.760, 0.765, 0.021, pr_auc=0.86)},
]


class TestMeetsTargets:
    def test_all_targets_met(self):
        assert meets_targets(row(0.988, 0.938, 0.824, 0.877, 0.005))

    def test_recall_just_under_target_fails(self):
        assert not meets_targets(row(0.988, 0.938, 0.819, 0.877, 0.005))

    def test_fpr_just_over_target_fails(self):
        assert not meets_targets(row(0.988, 0.938, 0.824, 0.877, 0.021))

    def test_targets_match_the_brief(self):
        assert TARGETS["auc_roc"] == (">=", 0.87)
        assert TARGETS["precision"] == (">=", 0.89)
        assert TARGETS["recall"] == (">=", 0.82)
        assert TARGETS["f1"] == (">=", 0.85)
        assert TARGETS["fpr"] == ("<=", 0.02)


class TestAggregate:
    def test_mean_is_correct(self):
        agg = aggregate(PER_SEED)
        expected = (0.988 + 0.981 + 0.990) / 3
        assert agg["GNN (GATV2)"]["auc_roc"][0] == pytest.approx(expected)

    def test_std_is_correct(self):
        agg = aggregate(PER_SEED)
        vals = [0.988, 0.981, 0.990]
        assert agg["GNN (GATV2)"]["auc_roc"][1] == pytest.approx(np.std(vals))

    def test_every_model_and_metric_present(self):
        agg = aggregate(PER_SEED)
        assert set(agg) == {"GNN (GATV2)", "LR", "Boost + neighbour averages"}
        for model in agg:
            for m in ("auc_roc", "pr_auc", "precision", "recall", "f1", "fpr"):
                assert m in agg[model]


class TestStrongestBaseline:
    def test_picks_highest_auc_baseline_excluding_gnn(self):
        agg = aggregate(PER_SEED)
        assert strongest_baseline(agg) == "Boost + neighbour averages"

    def test_returns_none_if_no_baseline_present(self):
        agg = aggregate([{"GNN (GATV2)": row(0.98, 0.9, 0.85, 0.87, 0.01)}])
        assert strongest_baseline(agg) is None


class TestPaired:
    def test_mean_difference_and_wins(self):
        p = paired(PER_SEED, "GNN (GATV2)", "Boost + neighbour averages", "auc_roc")
        expected_mean = np.mean([0.988 - 0.973, 0.981 - 0.965, 0.990 - 0.975])
        assert p["mean"] == pytest.approx(expected_mean)
        assert p["wins"] == 3
        assert p["n"] == 3

    def test_wins_counts_only_strict_improvements(self):
        # a tie (a - b == 0) is a well-defined edge case: it does NOT count as a win.
        tie_seed = [{"A": row(0.9, 0.9, 0.9, 0.9, 0.01), "B": row(0.9, 0.9, 0.9, 0.9, 0.01)}]
        p = paired(tie_seed, "A", "B", "auc_roc")
        assert p["wins"] == 0


class TestMarkdownTable:
    def test_contains_a_row_per_model(self):
        agg = aggregate(PER_SEED)
        md = markdown_table(agg, PER_SEED, {"accounts": 5000, "camouflage": 1.5})
        for model in agg:
            assert model in md

    def test_reports_how_many_seeds_meet_all_targets(self):
        agg = aggregate(PER_SEED)
        md = markdown_table(agg, PER_SEED, {"accounts": 5000, "camouflage": 1.5})
        # PER_SEED has exactly one seed (index 1) where the GNN's recall misses the target
        assert "met in 2 of 3 seeds" in md

    def test_includes_the_seed_count_and_graph_size(self):
        agg = aggregate(PER_SEED)
        md = markdown_table(agg, PER_SEED, {"accounts": 5000, "camouflage": 1.5})
        assert "3 seeds" in md
        assert "5,000 accounts" in md
