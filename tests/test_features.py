"""Tests for gnn_aml_core.features and gnn_aml_core.evaluation. Neither module needs PyTorch, so these tests
always run, even in an environment where torch is not installed."""
import numpy as np
import pytest

from gnn_aml_core.evaluation import best_f1_threshold, binary_metrics, full_report, rank_metrics
from gnn_aml_core.features import (EDGE_DIM, FEATURE_NAMES, NUM_RELATIONS, build_edge_features, build_node_features,
                                   make_bidirectional, merge_reverse_copies, standardize)


class TestBuildNodeFeatures:
    def test_shape_matches_feature_names(self, small_graph):
        x, names = build_node_features(small_graph.accounts, small_graph.transactions)
        assert x.shape == (len(small_graph.accounts), len(FEATURE_NAMES))
        assert names == FEATURE_NAMES

    def test_no_nan_or_inf(self, small_graph):
        x, _ = build_node_features(small_graph.accounts, small_graph.transactions)
        assert np.isfinite(x).all()

    def test_kyc_risk_matches_account_risk_score(self, small_graph):
        x, names = build_node_features(small_graph.accounts, small_graph.transactions)
        col = x[:, names.index("kyc_risk")]
        np.testing.assert_allclose(col, small_graph.accounts.risk_score.values, atol=1e-6)

    def test_is_offshore_matches_country_list(self, small_graph):
        from aml_synth.graph_generator import OFFSHORE
        x, names = build_node_features(small_graph.accounts, small_graph.transactions)
        col = x[:, names.index("is_offshore")]
        expected = small_graph.accounts.country.isin(OFFSHORE).astype(float).values
        np.testing.assert_allclose(col, expected)

    def test_account_type_one_hot_sums_to_one(self, small_graph):
        x, names = build_node_features(small_graph.accounts, small_graph.transactions)
        ix = [names.index(f"type_{t}") for t in ("individual", "business", "shell")]
        assert np.allclose(x[:, ix].sum(axis=1), 1.0)

    def test_degree_features_nonnegative(self, small_graph):
        x, names = build_node_features(small_graph.accounts, small_graph.transactions)
        for f in ("out_deg", "in_deg", "out_uniq", "in_uniq"):
            assert (x[:, names.index(f)] >= 0).all()


class TestStandardize:
    def test_output_has_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        x = rng.normal(loc=5, scale=3, size=(500, 4))
        xs, mean, std = standardize(x)
        assert np.allclose(xs.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(xs.std(axis=0), 1, atol=1e-6)

    def test_reapplying_given_mean_std_is_consistent(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 3))
        _, mean, std = standardize(x)
        x2 = rng.normal(size=(50, 3))
        xs2, mean2, std2 = standardize(x2, mean=mean, std=std)
        np.testing.assert_allclose(mean2, mean)
        np.testing.assert_allclose(xs2, (x2 - mean) / std)

    def test_constant_column_does_not_divide_by_zero(self):
        x = np.ones((10, 2))
        xs, _, _ = standardize(x)
        assert np.isfinite(xs).all()


class TestEdgeFeatures:
    def test_shape(self, small_graph):
        attr, etype = build_edge_features(small_graph.transactions)
        assert attr.shape == (len(small_graph.transactions), EDGE_DIM - 1)  # -1: direction flag added later by make_bidirectional
        assert etype.shape == (len(small_graph.transactions),)

    def test_relation_ids_in_range(self, small_graph):
        _, etype = build_edge_features(small_graph.transactions)
        assert etype.min() >= 0
        assert etype.max() < NUM_RELATIONS // 2  # base relations, before make_bidirectional doubles the range


class TestBidirectional:
    def test_doubles_edge_count(self):
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        attr = np.zeros((3, 4), dtype=np.float32)
        etype = np.array([0, 1, 0])
        s2, d2, a2, t2 = make_bidirectional(src, dst, attr, etype)
        assert len(s2) == len(d2) == 6
        assert a2.shape == (6, 5)

    def test_reversed_copies_swap_src_dst(self):
        src, dst = np.array([0, 1]), np.array([1, 2])
        attr = np.zeros((2, 2), dtype=np.float32)
        etype = np.array([0, 0])
        s2, d2, _, _ = make_bidirectional(src, dst, attr, etype)
        np.testing.assert_array_equal(s2[2:], dst)
        np.testing.assert_array_equal(d2[2:], src)

    def test_direction_flag_column(self):
        src, dst = np.array([0]), np.array([1])
        attr = np.zeros((1, 1), dtype=np.float32)
        etype = np.array([0])
        _, _, a2, _ = make_bidirectional(src, dst, attr, etype)
        assert a2[0, -1] == 0.0  # forward
        assert a2[1, -1] == 1.0  # reversed

    def test_relation_id_shifted_for_reverse(self):
        src, dst = np.array([0]), np.array([1])
        attr = np.zeros((1, 1), dtype=np.float32)
        etype = np.array([2])
        _, _, _, t2 = make_bidirectional(src, dst, attr, etype)
        from gnn_aml_core.features import RELATIONS_BASE
        assert t2[0] == 2
        assert t2[1] == 2 + RELATIONS_BASE


class TestMergeReverseCopies:
    def test_keeps_larger_importance_and_true_direction(self):
        # tx 2 appears as both a forward edge (id=2) and a reversed copy (id=2+n_tx); the reversed one is more
        # important, so its (swapped-back) direction should win.
        n_tx = 10
        ids = np.array([2, 2 + n_tx, 5])
        s_l = np.array([0, 3, 1])
        d_l = np.array([3, 0, 2])
        imp = np.array([0.3, 0.9, 0.2])
        out = merge_reverse_copies(ids, s_l, d_l, imp, n_tx=n_tx, top_k=10)
        assert len(out) == 2
        tx2 = next(r for r in out if r[0] == 2)
        assert tx2[1:3] == (0, 3)  # direction restored to the ORIGINAL sense, not the reversed copy's local order
        assert tx2[3] == 0.9

    def test_respects_top_k(self):
        n_tx = 5
        ids = np.arange(n_tx)
        out = merge_reverse_copies(ids, ids, ids + 1, np.linspace(0, 1, n_tx), n_tx=n_tx, top_k=2)
        assert len(out) == 2

    def test_sorted_by_importance_descending(self):
        n_tx = 4
        ids = np.arange(n_tx)
        imp = np.array([0.1, 0.9, 0.5, 0.3])
        out = merge_reverse_copies(ids, ids, ids + 1, imp, n_tx=n_tx, top_k=10)
        assert [r[3] for r in out] == sorted([r[3] for r in out], reverse=True)


class TestEvaluationMetrics:
    def test_rank_metrics_perfect_separation(self):
        y = [0, 0, 0, 1, 1, 1]
        p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        m = rank_metrics(y, p)
        assert m["auc_roc"] == 1.0
        assert m["pr_auc"] == 1.0

    def test_rank_metrics_random_guess(self):
        y = [0, 1, 0, 1]
        p = [0.5, 0.5, 0.5, 0.5]
        m = rank_metrics(y, p)
        assert 0.0 <= m["auc_roc"] <= 1.0

    def test_binary_metrics_hand_computed(self):
        y = [1, 1, 0, 0, 0, 0]
        p = [0.9, 0.4, 0.8, 0.1, 0.2, 0.3]
        m = binary_metrics(y, p, threshold=0.5)
        assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
        assert m["precision"] == pytest.approx(0.5)
        assert m["recall"] == pytest.approx(0.5)
        assert m["fpr"] == pytest.approx(0.25)
        assert m["f1"] == pytest.approx(0.5)

    def test_binary_metrics_no_positives_predicted(self):
        m = binary_metrics([1, 0, 0], [0.1, 0.1, 0.1], threshold=0.9)
        assert m["tp"] == 0 and m["precision"] == 0.0

    def test_best_f1_threshold_finds_the_separating_point(self):
        y = [0, 0, 0, 1, 1, 1]
        p = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        thr = best_f1_threshold(y, p)
        preds = [x >= thr for x in p]
        assert preds == [False, False, False, True, True, True]

    def test_full_report_threshold_from_validation_applied_to_test(self):
        y_val = [0, 0, 1, 1]
        p_val = [0.1, 0.4, 0.6, 0.9]
        y_test = [0, 1]
        p_test = [0.3, 0.7]
        rep = full_report(y_val, p_val, y_test, p_test)
        assert rep["threshold"] == best_f1_threshold(y_val, p_val)
        assert "auc_roc" in rep and "f1" in rep and "fpr" in rep
