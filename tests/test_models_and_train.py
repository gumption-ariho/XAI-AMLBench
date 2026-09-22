"""Tests for gnn_aml_core.models and gnn_aml_core.train. These need PyTorch and torch_geometric; the whole module
is skipped (not failed) when they are not installed, so `pytest` still runs cleanly on a machine without them."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnn_aml_core.features import build_edge_features, make_bidirectional
from gnn_aml_core.models import GATv2Detector, RGCNDetector, build_model, class_weighted_bce
from gnn_aml_core.train import arrays_from_frames, fit_gnn, make_splits


@pytest.fixture(scope="module")
def arrays(small_graph):
    return arrays_from_frames(small_graph.accounts, small_graph.transactions)


class TestBuildModel:
    @pytest.mark.parametrize("name,cls", [("gatv2", GATv2Detector), ("rgcn", RGCNDetector)])
    def test_returns_correct_class(self, name, cls):
        m = build_model(name, in_dim=10)
        assert isinstance(m, cls)

    def test_unknown_model_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            build_model("not_a_real_model", in_dim=10)

    def test_forward_pass_shape(self, arrays):
        x = torch.from_numpy(arrays["x"])
        ei = torch.from_numpy(arrays["edge_index"])
        ea = torch.from_numpy(arrays["edge_attr"])
        et = torch.from_numpy(arrays["edge_type"])
        model = build_model("gatv2", in_dim=x.shape[1])
        out = model(x, ei, ea, et)
        assert out.shape == (x.shape[0],)

    def test_output_has_no_nan(self, arrays):
        x = torch.from_numpy(arrays["x"])
        ei = torch.from_numpy(arrays["edge_index"])
        ea = torch.from_numpy(arrays["edge_attr"])
        et = torch.from_numpy(arrays["edge_type"])
        model = build_model("gatv2", in_dim=x.shape[1])
        out = model(x, ei, ea, et)
        assert not torch.isnan(out).any()


class TestClassWeightedBCE:
    def test_default_weight_is_neg_over_pos(self):
        logits = torch.zeros(10)
        targets = torch.cat([torch.zeros(8), torch.ones(2)])
        loss_default = class_weighted_bce(logits, targets)
        loss_explicit = class_weighted_bce(logits, targets, pos_weight=4.0)  # 8 neg / 2 pos = 4.0
        assert torch.isclose(loss_default, loss_explicit, atol=1e-4)

    def test_higher_pos_weight_penalises_missed_positives_more(self):
        logits = torch.tensor([-5.0, -5.0])  # confidently predicts "negative" for both
        targets = torch.tensor([1.0, 1.0])   # both are actually positive: a bad miss
        low = class_weighted_bce(logits, targets, pos_weight=1.0)
        high = class_weighted_bce(logits, targets, pos_weight=10.0)
        assert high > low

    def test_loss_is_finite(self):
        logits = torch.randn(20)
        targets = (torch.rand(20) > 0.5).float()
        loss = class_weighted_bce(logits, targets)
        assert torch.isfinite(loss)


class TestArraysFromFrames:
    def test_keys_present(self, arrays):
        for key in ("x", "edge_index", "edge_attr", "edge_type", "y", "n_tx", "account_ids", "feature_names"):
            assert key in arrays

    def test_edge_index_is_bidirectional_of_transactions(self, arrays):
        n_tx = arrays["n_tx"]
        assert arrays["edge_index"].shape[1] == 2 * n_tx

    def test_labels_match_account_count(self, arrays):
        assert len(arrays["y"]) == arrays["x"].shape[0]


class TestMakeSplits:
    def test_splits_are_disjoint_and_cover_everything(self, arrays):
        tr, va, te = make_splits(arrays["y"], seed=0)
        all_idx = np.concatenate([tr, va, te])
        assert len(set(all_idx)) == len(arrays["y"])
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))

    def test_roughly_60_20_20(self, arrays):
        tr, va, te = make_splits(arrays["y"], seed=0)
        n = len(arrays["y"])
        assert abs(len(tr) / n - 0.6) < 0.02
        assert abs(len(va) / n - 0.2) < 0.02
        assert abs(len(te) / n - 0.2) < 0.02

    def test_stratified_by_label_rate(self, arrays):
        tr, va, te = make_splits(arrays["y"], seed=0)
        y = arrays["y"]
        overall = y.mean()
        for split in (tr, va, te):
            assert abs(y[split].mean() - overall) < 0.03

    def test_same_seed_reproducible(self, arrays):
        a1 = make_splits(arrays["y"], seed=5)
        a2 = make_splits(arrays["y"], seed=5)
        for x1, x2 in zip(a1, a2):
            np.testing.assert_array_equal(x1, x2)


class TestFitGnn:
    def test_short_run_produces_a_report(self, arrays):
        tr, va, te = make_splits(arrays["y"], seed=0)
        res = fit_gnn(arrays, tr, va, te, model_name="gatv2", epochs=6, eval_every=3, seed=0, verbose=False)
        rep = res["report"]
        for key in ("auc_roc", "precision", "recall", "f1", "fpr"):
            assert key in rep
            assert 0.0 <= rep[key] <= 1.0

    def test_patience_stops_early(self, arrays):
        tr, va, te = make_splits(arrays["y"], seed=0)
        res = fit_gnn(arrays, tr, va, te, model_name="gatv2", epochs=200, eval_every=5, patience=1, seed=0, verbose=False)
        assert res["epochs_run"] < 200
