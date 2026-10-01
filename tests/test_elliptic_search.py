"""Tests for gnn_aml_core.elliptic_search. search_gnn's model-selection logic is tested with a mocked fit_gnn
(no torch needed); embeddings_plus_boosting needs torch and is skipped if unavailable."""
import numpy as np
import pytest


class TestSearchGnn:
    def test_selects_the_config_with_the_best_validation_auc(self, monkeypatch):
        """The critical correctness property: selection must be by VALIDATION auc only, never by test score --
        using test performance to pick a hyperparameter configuration would be test-set leakage. This is
        checked by deliberately inverting val and test scores across configs, so picking by the wrong metric
        would select the wrong config."""
        calls = []

        def fake_fit_gnn(arrays, tr, va, te, **kw):
            calls.append(kw)
            val_auc = {64: 0.80, 128: 0.90, 96: 0.85}[kw["hidden"]]
            fake_test_auc = 1.0 - val_auc  # deliberately inverted vs val_auc
            return {"report": {"auc_roc": fake_test_auc, "f1": 0.5}, "best_val_auc": val_auc, "best_epoch": 1,
                   "epochs_run": 1, "state_dict": {}, "hparams": kw}

        import gnn_aml_core.train as train_module
        monkeypatch.setattr(train_module, "fit_gnn", fake_fit_gnn)

        from gnn_aml_core.elliptic_search import search_gnn
        grid = [{"hidden": 64, "layers": 2, "lr": 0.005}, {"hidden": 128, "layers": 2, "lr": 0.005},
               {"hidden": 96, "layers": 2, "lr": 0.01}]
        arrays = {"edge_attr": np.zeros((10, 5))}
        best = search_gnn(arrays, [0], [1], [2], "gatv2", grid, epochs=1, patience=1, seed=0)

        assert best["best_val_auc"] == 0.90
        assert best["hp"]["hidden"] == 128
        assert len(calls) == 3

    def test_tries_every_configuration_in_the_grid_exactly_once(self, monkeypatch):
        calls = []

        def fake_fit_gnn(arrays, tr, va, te, **kw):
            calls.append((kw["hidden"], kw["layers"]))
            return {"report": {"auc_roc": 0.5, "f1": 0.5}, "best_val_auc": 0.5, "best_epoch": 1,
                   "epochs_run": 1, "state_dict": {}, "hparams": kw}

        import gnn_aml_core.train as train_module
        monkeypatch.setattr(train_module, "fit_gnn", fake_fit_gnn)

        from gnn_aml_core.elliptic_search import GATV2_GRID, search_gnn
        arrays = {"edge_attr": np.zeros((10, 5))}
        search_gnn(arrays, [0], [1], [2], "gatv2", GATV2_GRID, epochs=1, patience=1, seed=0)
        assert len(calls) == len(GATV2_GRID)
        assert len(set(calls)) == len(GATV2_GRID), "each grid configuration should be tried, not repeated"

    def test_rgcn_does_not_pass_edge_dim_kwarg(self, monkeypatch):
        # GATv2 needs edge_dim; RGCN's build_model signature does not accept it -- passing it anyway would
        # raise a TypeError inside the real fit_gnn, so this is checked directly.
        received_kwargs = {}

        def fake_fit_gnn(arrays, tr, va, te, **kw):
            received_kwargs.update(kw)
            return {"report": {"auc_roc": 0.5, "f1": 0.5}, "best_val_auc": 0.5, "best_epoch": 1,
                   "epochs_run": 1, "state_dict": {}, "hparams": kw}

        import gnn_aml_core.train as train_module
        monkeypatch.setattr(train_module, "fit_gnn", fake_fit_gnn)

        from gnn_aml_core.elliptic_search import RGCN_GRID, search_gnn
        arrays = {"edge_attr": np.zeros((10, 5))}
        search_gnn(arrays, [0], [1], [2], "rgcn", RGCN_GRID[:1], epochs=1, patience=1, seed=0)
        assert "edge_dim" not in received_kwargs


class TestEmbeddingsPlusBoosting:
    def test_returns_a_report_keyed_by_model_name(self):
        torch = pytest.importorskip("torch")
        from gnn_aml_core.elliptic_search import embeddings_plus_boosting
        from gnn_aml_core.models import build_model

        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(size=(n, 6)).astype(np.float32)
        edge_index = np.array([rng.integers(0, n, 300), rng.integers(0, n, 300)])
        edge_attr = rng.normal(size=(300, 5)).astype(np.float32)
        y = (rng.random(n) < 0.2).astype(np.float32)
        arrays = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "y": y}

        model = build_model("gatv2", in_dim=6, hidden=8, num_layers=2, edge_dim=5)
        tr = np.arange(0, 120)
        va = np.arange(120, 160)
        te = np.arange(160, 200)
        best_gnn = {"hp": {"hidden": 8, "layers": 2, "lr": 0.01}, "state_dict": model.state_dict()}

        result = embeddings_plus_boosting(arrays, best_gnn, "gatv2", tr, va, te, seed=0)
        assert len(result) == 1
        key = next(iter(result))
        assert "GATV2" in key
        assert "auc_roc" in result[key]
