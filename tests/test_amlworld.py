"""Tests for gnn_aml_core.amlworld. The fixture deliberately writes "Account" as a column header TWICE (not
pre-renamed to "Account.1"), replicating the real file's actual raw header exactly as confirmed by direct
inspection of documented real AMLworld write-ups, not assumed."""
from pathlib import Path

import numpy as np
import pytest

from gnn_aml_core.amlworld import load_amlworld


def _write_fixture(tmp_path: Path) -> Path:
    p = tmp_path / "LI-Small_Trans.csv"
    p.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,"
        "Payment Currency,Payment Format,Is Laundering\n"
        "2022-01-01 00:00:00,10,ACC001,20,ACC002,100.0,USD,100.0,USD,ACH,0\n"
        "2022-01-01 01:00:00,20,ACC002,10,ACC001,50.0,USD,50.0,USD,ACH,0\n"
        "2022-01-02 00:00:00,10,ACC003,30,ACC004,500.0,USD,500.0,USD,Wire,1\n"
        "2022-01-02 01:00:00,30,ACC004,40,ACC005,500.0,USD,500.0,USD,Wire,1\n"
        "2022-01-03 00:00:00,10,ACC001,30,ACC004,20.0,USD,20.0,USD,ACH,0\n"
    )
    return p


class TestLoadAmlworld:
    def test_missing_file_raises_a_clear_error_not_a_generic_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="kaggle.com"):
            load_amlworld(tmp_path / "does_not_exist.csv")

    def test_the_duplicate_account_header_is_parsed_correctly(self, tmp_path):
        # confirms pandas' own auto-rename (Account -> Account, Account.1) is handled, not assumed
        accounts, tx = load_amlworld(_write_fixture(tmp_path))
        assert len(tx) == 5
        assert "10_ACC001" in accounts["account_id"].values
        assert "40_ACC005" in accounts["account_id"].values

    def test_account_ids_combine_bank_and_account_for_global_uniqueness(self, tmp_path):
        accounts, tx = load_amlworld(_write_fixture(tmp_path))
        # ACC001 appears under bank 10 only in this fixture, but the id must still carry the bank, not just
        # the raw account number, since the same account number can recur under a different bank in the real
        # ~30,000-bank dataset
        assert all("_" in a for a in accounts["account_id"])

    def test_label_aggregation_flags_only_accounts_in_an_actually_flagged_transaction(self, tmp_path):
        # tx_4 (10_ACC001 -> 30_ACC004) is itself NOT flagged, even though 30_ACC004 IS flagged via a
        # DIFFERENT transaction -- 10_ACC001 must not be flagged just for having transacted with a flagged
        # account; only participating in an actually-flagged transaction should flag an account
        accounts, tx = load_amlworld(_write_fixture(tmp_path))
        by_id = accounts.set_index("account_id")["is_suspicious"]
        assert by_id["10_ACC001"] == 0.0
        assert by_id["20_ACC002"] == 0.0
        assert by_id["10_ACC003"] == 1.0
        assert by_id["30_ACC004"] == 1.0
        assert by_id["40_ACC005"] == 1.0

    def test_cross_border_proxy_is_different_bank_not_different_country(self, tmp_path):
        accounts, tx = load_amlworld(_write_fixture(tmp_path))
        # every transaction in this fixture happens to be between different banks
        assert (tx["cross_border"] == 1).all()

    def test_output_feeds_arrays_from_frames_without_nan_or_inf(self, tmp_path):
        from gnn_aml_core.train import arrays_from_frames

        accounts, tx = load_amlworld(_write_fixture(tmp_path))
        arrays = arrays_from_frames(accounts, tx.drop(columns=["is_laundering"]))
        assert not np.isnan(arrays["x"]).any()
        assert not np.isinf(arrays["x"]).any()
        assert arrays["y"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]

    def test_missing_required_columns_raises_a_clear_error(self, tmp_path):
        p = tmp_path / "LI-Small_Trans.csv"
        p.write_text("Timestamp,From Bank,Account\n2022-01-01,10,ACC001\n")
        with pytest.raises(ValueError, match="missing"):
            load_amlworld(p)


class TestMain:
    """main()'s own control flow: argument wiring, the --save/--no-baselines branches, and the
    meets_auc_target/metrics computation -- with load_amlworld's real output but fit_gnn/evaluate_baselines/
    save_serving_artifacts all mocked out, since actually training a real GATv2/RGCN model needs torch and
    torch_geometric and belongs to gnn_aml_core.train's own test suite, not this module's argument-wiring
    logic."""

    def _install_mocks(self, monkeypatch, *, fit_gnn_result=None):
        import gnn_aml_core.amlworld as aw

        fake_arrays = {"x": np.zeros((3, 2)), "y": np.array([0.0, 1.0, 0.0]), "src_tx": np.array([0, 1]),
                      "dst_tx": np.array([1, 2]), "edge_attr": np.zeros((2, 4)), "n_tx": 2}

        import gnn_aml_core.train as train_mod
        monkeypatch.setattr(train_mod, "arrays_from_frames", lambda accounts, tx: fake_arrays)
        monkeypatch.setattr(train_mod, "make_splits", lambda y, seed=42: ([0], [1], [2]))

        default_report = {"auc_roc": 0.91, "pr_auc": 0.5, "precision": 0.6, "recall": 0.7, "f1": 0.65,
                          "fpr": 0.02, "threshold": 0.5}
        fit_gnn_result = fit_gnn_result or {"report": default_report, "best_epoch": 42, "best_val_auc": 0.9,
                                            "hparams": {"hidden": 64}, "calibration_report": {"ece": 0.01}}
        monkeypatch.setattr(train_mod, "fit_gnn", lambda *a, **kw: fit_gnn_result)
        monkeypatch.setattr(train_mod, "format_table", lambda table: "FORMATTED_TABLE")

        save_calls = []
        monkeypatch.setattr(train_mod, "save_serving_artifacts",
                           lambda *a, **kw: save_calls.append((a, kw)))

        baseline_calls = []
        import gnn_aml_core.baselines as baselines_mod
        monkeypatch.setattr(baselines_mod, "evaluate_baselines",
                           lambda *a, **kw: baseline_calls.append((a, kw)) or {"Logistic Regression": default_report})

        return save_calls, baseline_calls

    def test_evaluation_only_run_does_not_save_and_logs_the_evaluation_only_notice(self, monkeypatch, tmp_path, caplog):
        import logging
        import sys

        import gnn_aml_core.amlworld as aw

        save_calls, _ = self._install_mocks(monkeypatch)
        _write_fixture(tmp_path)
        monkeypatch.setattr(sys, "argv", ["amlworld.py", "--data", str(tmp_path)])

        with caplog.at_level(logging.INFO, logger="gnn_aml_core.amlworld"):
            aw.main()

        assert save_calls == []  # --save was not given -- nothing should be written
        assert "evaluation only (no --save given)" in caplog.text

    def test_no_baselines_flag_skips_the_baseline_evaluation(self, monkeypatch, tmp_path):
        import sys

        import gnn_aml_core.amlworld as aw

        _, baseline_calls = self._install_mocks(monkeypatch)
        _write_fixture(tmp_path)
        monkeypatch.setattr(sys, "argv", ["amlworld.py", "--data", str(tmp_path), "--no-baselines"])

        aw.main()
        assert baseline_calls == []

    def test_save_flag_writes_serving_artifacts_with_computed_metrics(self, monkeypatch, tmp_path):
        import sys

        import gnn_aml_core.amlworld as aw

        save_calls, _ = self._install_mocks(monkeypatch)
        _write_fixture(tmp_path)
        out_dir = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", ["amlworld.py", "--data", str(tmp_path), "--save", "--out", str(out_dir),
                                         "--auc-target", "0.85"])

        aw.main()

        assert len(save_calls) == 1
        _, kwargs = save_calls[0]
        metrics = kwargs["extra_metrics"] if "extra_metrics" in kwargs else None
        # metrics passed as the positional `metrics` dict (3rd positional arg) carries meets_auc_target
        args = save_calls[0][0]
        passed_metrics = args[2]
        assert passed_metrics["meets_auc_target"] is True  # 0.91 (mocked report) >= 0.85
        assert passed_metrics["auc_target"] == 0.85
        assert passed_metrics["test_auc_roc"] == 0.91

    def test_meets_auc_target_is_false_when_the_real_auc_falls_short(self, monkeypatch, tmp_path):
        import sys

        import gnn_aml_core.amlworld as aw

        low_report = {"auc_roc": 0.80, "pr_auc": 0.5, "precision": 0.6, "recall": 0.7, "f1": 0.65,
                     "fpr": 0.02, "threshold": 0.5}
        save_calls, _ = self._install_mocks(monkeypatch, fit_gnn_result={
            "report": low_report, "best_epoch": 10, "best_val_auc": 0.8,
            "hparams": {"hidden": 32}, "calibration_report": {"ece": 0.02}})
        _write_fixture(tmp_path)
        monkeypatch.setattr(sys, "argv", ["amlworld.py", "--data", str(tmp_path), "--save",
                                         "--out", str(tmp_path / "out"), "--auc-target", "0.87"])

        aw.main()
        passed_metrics = save_calls[0][0][2]
        assert passed_metrics["meets_auc_target"] is False

    def test_model_choice_is_threaded_through_to_fit_gnn_kwargs(self, monkeypatch, tmp_path):
        import sys

        import gnn_aml_core.amlworld as aw
        import gnn_aml_core.train as train_mod

        self._install_mocks(monkeypatch)
        captured = {}

        def fake_fit_gnn(arrays, tr, va, te, **kw):
            captured.update(kw)
            return {"report": {"auc_roc": 0.9, "pr_auc": 0.5, "precision": 0.6, "recall": 0.7, "f1": 0.6,
                              "fpr": 0.02, "threshold": 0.5}, "best_epoch": 1, "best_val_auc": 0.9,
                   "hparams": {}, "calibration_report": {}}

        monkeypatch.setattr(train_mod, "fit_gnn", fake_fit_gnn)
        _write_fixture(tmp_path)
        monkeypatch.setattr(sys, "argv", ["amlworld.py", "--data", str(tmp_path), "--model", "rgcn"])

        aw.main()
        assert captured["model_name"] == "rgcn"
        assert "edge_dim" not in captured  # edge_dim is only passed for gatv2, never for rgcn
