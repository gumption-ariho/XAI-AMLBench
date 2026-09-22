"""Tests for gnn_aml_core.baselines. Torch-free (scikit-learn only), so these always run."""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from gnn_aml_core.baselines import evaluate_baselines, neighbour_aggregates
from gnn_aml_core.features import build_node_features, standardize


@pytest.fixture(scope="module")
def features_and_splits(small_graph):
    x, _ = build_node_features(small_graph.accounts, small_graph.transactions)
    y = small_graph.accounts.is_suspicious.to_numpy()
    xs, _, _ = standardize(x)
    idx = {a: i for i, a in enumerate(small_graph.accounts.account_id)}
    src = small_graph.transactions.src.map(idx).to_numpy()
    dst = small_graph.transactions.dst.map(idx).to_numpy()
    tr, tmp = train_test_split(np.arange(len(y)), test_size=0.4, random_state=0, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=0, stratify=y[tmp])
    return xs, y, tr, va, te, src, dst


class TestNeighbourAggregates:
    def test_output_shape_is_own_plus_hop_features(self):
        x = np.arange(12, dtype=float).reshape(4, 3)
        src, dst = np.array([0, 1, 2]), np.array([1, 2, 3])
        out = neighbour_aggregates(x, src, dst, hops=2)
        assert out.shape == (4, 3 * 3)  # own + 1-hop + 2-hop

    def test_isolated_node_gets_zero_neighbour_average(self):
        x = np.array([[1.0], [2.0], [3.0]])
        src, dst = np.array([0]), np.array([1])  # node 2 has no edges at all
        out = neighbour_aggregates(x, src, dst, hops=1)
        assert out[2, 1] == 0.0  # its neighbour-average column


class TestEvaluateBaselines:
    def test_returns_four_models(self, features_and_splits):
        xs, y, tr, va, te, src, dst = features_and_splits
        out = evaluate_baselines(xs, y, tr, va, te, edges=(src, dst), seed=0)
        assert len(out) == 4
        assert any("Logistic Regression" in k for k in out)
        assert any("Isolation Forest" in k for k in out)
        assert any("neighbour averages" in k for k in out)

    def test_without_edges_returns_three_models(self, features_and_splits):
        xs, y, tr, va, te, _, _ = features_and_splits
        out = evaluate_baselines(xs, y, tr, va, te, edges=None, seed=0)
        assert len(out) == 3

    def test_every_model_reports_full_metric_set(self, features_and_splits):
        xs, y, tr, va, te, src, dst = features_and_splits
        out = evaluate_baselines(xs, y, tr, va, te, edges=(src, dst), seed=0)
        for name, rep in out.items():
            for key in ("auc_roc", "pr_auc", "precision", "recall", "f1", "fpr", "threshold"):
                assert key in rep, f"{name} is missing {key}"

    def test_metrics_are_valid_probabilities_or_rates(self, features_and_splits):
        xs, y, tr, va, te, src, dst = features_and_splits
        out = evaluate_baselines(xs, y, tr, va, te, edges=(src, dst), seed=0)
        for rep in out.values():
            for key in ("auc_roc", "pr_auc", "precision", "recall", "f1", "fpr"):
                assert 0.0 <= rep[key] <= 1.0

    def test_graph_aware_baseline_beats_or_matches_plain_boosting_on_easy_case(self, features_and_splits):
        # Sanity direction check, not a strict AUC bound: a model that also sees neighbour features should not be
        # dramatically worse than the plain model on the same data.
        xs, y, tr, va, te, src, dst = features_and_splits
        out = evaluate_baselines(xs, y, tr, va, te, edges=(src, dst), seed=0)
        plain = next(v for k, v in out.items() if "neighbour" not in k and "Logistic" not in k and "Isolation" not in k)
        graph = next(v for k, v in out.items() if "neighbour" in k)
        assert graph["auc_roc"] >= plain["auc_roc"] - 0.15
