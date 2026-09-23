"""Tests for xai_explainer.extractor. Needs PyTorch and torch_geometric; skipped (not failed) when they are not
installed. Uses an untrained GATv2Detector: GNNExplainer's masks are meaningless on an untrained model, but every
test here checks the extraction's STRUCTURE (return shape, error handling, which edges/nodes come back, that the
graph's bidirectionality is folded correctly) rather than prediction quality, so training is not needed."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnn_aml_core.models import build_model
from gnn_aml_core.train import arrays_from_frames
from xai_explainer.extractor import extract_explanation

EPOCHS = 5  # GNNExplainer epochs; kept low so the test suite stays fast (a real /explain call uses more)


@pytest.fixture(scope="module")
def arrays(small_graph):
    return arrays_from_frames(small_graph.accounts, small_graph.transactions)


@pytest.fixture(scope="module")
def model(arrays):
    torch.manual_seed(0)
    return build_model("gatv2", in_dim=arrays["x"].shape[1])


def explain(model, arrays, account_id, **kw):
    kw.setdefault("hops", 2)
    kw.setdefault("reporting_threshold", 10_000)
    kw.setdefault("epochs", EPOCHS)
    return extract_explanation(model, arrays, arrays["feature_names"], account_id, **kw)


class TestExtractExplanation:
    def test_unknown_account_raises_keyerror(self, model, arrays):
        with pytest.raises(KeyError):
            explain(model, arrays, "NOT_A_REAL_ACCOUNT_ID")

    def test_returns_the_expected_keys(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        for key in ("account_id", "risk_score", "reporting_threshold", "nodes", "edges", "top_features"):
            assert key in result

    def test_account_id_echoed_back(self, model, arrays):
        account_id = arrays["account_ids"][5]
        result = explain(model, arrays, account_id)
        assert result["account_id"] == account_id

    def test_risk_score_is_a_probability(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_reporting_threshold_is_passed_through_unchanged(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0], reporting_threshold=12_345)
        assert result["reporting_threshold"] == 12_345

    def test_center_account_is_among_the_returned_nodes(self, model, arrays):
        account_id = arrays["account_ids"][3]
        result = explain(model, arrays, account_id)
        assert any(n["account_id"] == account_id for n in result["nodes"])

    def test_node_dicts_have_the_expected_shape(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        for n in result["nodes"]:
            assert set(n) == {"account_id", "account_type", "country"}

    def test_edge_dicts_have_the_expected_shape(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        for e in result["edges"]:
            assert set(e) == {"tx_id", "src", "dst", "amount", "timestamp", "cross_border", "importance"}

    def test_edges_never_exceed_top_k(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0], top_k=5)
        assert len(result["edges"]) <= 5

    def test_edge_endpoints_are_real_accounts(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        ids = set(arrays["account_ids"])
        for e in result["edges"]:
            assert e["src"] in ids and e["dst"] in ids

    def test_top_features_reference_real_feature_names(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        names = set(arrays["feature_names"])
        for f in result["top_features"]:
            assert f["feature"] in names

    def test_top_features_capped_at_six(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        assert len(result["top_features"]) <= 6

    def test_top_feature_weights_are_non_negative_shares(self, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        for f in result["top_features"]:
            assert 0.0 <= f["weight"] <= 1.0

    def test_top_feature_weights_sum_to_approximately_one(self, model, arrays):
        # Weights are normalised against the sum of the SHOWN features only (not all 26), so a fair, sparse
        # explanation always reads as a complete picture rather than being diluted by unshown features -- see
        # xai_explainer/extractor.py's comment on this for why the earlier normalisation produced flat-looking
        # bars (e.g. six features all around 4-6%) even when the underlying explanation was reasonably sparse.
        result = explain(model, arrays, arrays["account_ids"][0])
        total = sum(f["weight"] for f in result["top_features"])
        assert total == pytest.approx(1.0, abs=0.01)

    def test_small_max_explain_edges_forces_the_1_hop_fallback_without_erroring(self, model, arrays):
        # A max_explain_edges of 1 forces the "very large hub" fallback path on virtually any account; this just
        # confirms that path still returns a well-formed result rather than crashing.
        result = explain(model, arrays, arrays["account_ids"][0], max_explain_edges=1)
        for key in ("account_id", "nodes", "edges"):
            assert key in result

    def test_transactions_edge_amounts_come_from_the_original_data(self, small_graph, model, arrays):
        result = explain(model, arrays, arrays["account_ids"][0])
        real_amounts = {r.tx_id: float(r.amount) for r in small_graph.transactions.itertuples()}
        for e in result["edges"]:
            assert abs(e["amount"] - real_amounts[e["tx_id"]]) < 0.01
