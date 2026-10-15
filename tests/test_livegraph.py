"""Tests for gnn_aml_core.livegraph's live (non-synthetic) transaction store -- ingest()/read_live(),
remove_accounts() (the backing function for POST /admin/remove-live / backend's POST /intake/clear, the Intake
tab's "clear temporary data" control), _accounts_from_tx(), and the PyTorch-dependent
write_blank_graph()/rebuild_live_graph()/main() (skipped automatically, via pytest.importorskip, on a machine
without PyTorch installed).
"""
import os

os.environ.setdefault("LIVE_DATA_DIR", "data/live")   # read once at import time by the module under test

import numpy as np
import pandas as pd
import pytest

import gnn_aml_core.livegraph as lg


def _use_tmp_store(monkeypatch, tmp_path):
    """Points the module's live transaction store at an empty temp dir for this test, instead of whatever
    data/live resolved to at import time -- tests must never read or write the real on-disk live store."""
    monkeypatch.setattr(lg, "LIVE_DIR", tmp_path)


class TestIngestAndReadLive:
    def test_read_live_is_empty_before_anything_is_ingested(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        df = lg.read_live()
        assert len(df) == 0
        assert list(df.columns) == lg.TX_COLUMNS

    def test_ingest_is_additive_across_calls(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}])
        lg.ingest([{"src": "B", "dst": "C", "amount": 50}])
        df = lg.read_live()
        assert len(df) == 2

    def test_a_row_missing_src_or_dst_is_dropped_not_raised(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        result = lg.ingest([{"src": "", "dst": "B", "amount": 100}, {"src": "X", "dst": "Y", "amount": 10}])
        assert len(result) == 1
        assert result.iloc[0]["src"] == "X"

    def test_a_document_of_entirely_malformed_rows_is_a_no_op_and_never_touches_disk(self, monkeypatch, tmp_path):
        # every row is malformed (blank src/dst) -- new_rows ends up empty, so ingest() must return the
        # existing table untouched and must not write an (empty) file, rather than silently creating one
        _use_tmp_store(monkeypatch, tmp_path)
        result = lg.ingest([{"src": "", "dst": "", "amount": 1}, {"src": "", "dst": "B", "amount": 2}])
        assert len(result) == 0
        assert not lg._live_tx_path().exists()

    def test_ingest_fills_in_timestamp_and_cross_border_defaults(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        result = lg.ingest([{"src": "A", "dst": "B", "amount": 10}])  # no timestamp/cross_border given
        row = result.iloc[0]
        assert row["cross_border"] == 0
        assert row["timestamp"] > 0  # defaulted to "now", not left blank or zero

    def test_ingest_honors_an_explicit_timestamp_and_cross_border(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        result = lg.ingest([{"src": "A", "dst": "B", "amount": 10, "timestamp": 1700000000, "cross_border": 1}])
        row = result.iloc[0]
        assert row["timestamp"] == 1700000000
        assert row["cross_border"] == 1


class TestAccountsFromTx:
    def test_builds_one_row_per_unique_account_seen_as_either_src_or_dst(self):
        tx = pd.DataFrame({"src": ["A", "B", "A"], "dst": ["B", "C", "C"]})
        accounts = lg._accounts_from_tx(tx)
        assert set(accounts["account_id"]) == {"A", "B", "C"}
        assert len(accounts) == 3  # no duplicates even though A and B each appear more than once

    def test_uses_the_same_honest_placeholder_fields_amlworld_uses(self):
        # a raw transaction document carries no account_type/country/opened_ts/risk_score -- these must be
        # filled with the documented placeholder values, not left missing or guessed
        tx = pd.DataFrame({"src": ["A"], "dst": ["B"]})
        accounts = lg._accounts_from_tx(tx)
        assert set(accounts["account_type"]) == {"unknown"}
        assert set(accounts["country"]) == {"unknown"}
        assert (accounts["opened_ts"] == 0).all()
        assert (accounts["risk_score"] == 0.0).all()


torch = pytest.importorskip("torch")


def _fake_ckpt(feature_names):
    n = len(feature_names)
    return {"feature_names": feature_names, "mean": torch.zeros(n), "std": torch.from_numpy(np.ones(n))}


class TestWriteBlankGraph:
    def test_writes_a_zero_account_graph_matching_the_checkpoints_feature_names(self, tmp_path):
        ckpt = _fake_ckpt(["f1", "f2", "f3"])
        result = lg.write_blank_graph(ckpt, tmp_path)
        assert result == {"n_accounts": 0, "n_transactions": 0}
        graph = torch.load(tmp_path / "graph.pt", map_location="cpu", weights_only=False)
        assert graph["account_ids"] == []
        assert graph["feature_names"] == ["f1", "f2", "f3"]
        assert graph["x"].shape == (0, 3)

    def test_never_touches_gnn_model_pt(self, tmp_path):
        # write_blank_graph must only ever write graph.pt -- the trained weights file must not exist, let
        # alone be modified, as a result of calling it
        ckpt = _fake_ckpt(["f1"])
        lg.write_blank_graph(ckpt, tmp_path)
        assert not (tmp_path / "gnn_model.pt").exists()


class TestRebuildLiveGraph:
    def _real_feature_names(self, accounts, tx):
        from gnn_aml_core.features import build_node_features
        _, names = build_node_features(accounts, tx)
        return names

    def test_empty_live_store_writes_a_blank_graph_instead(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path / "live")
        accounts = lg._accounts_from_tx(pd.DataFrame({"src": ["A"], "dst": ["B"]}))
        names = self._real_feature_names(accounts, pd.DataFrame({"src": ["A"], "dst": ["B"], "amount": [1.0],
                                                                  "timestamp": [1], "cross_border": [0]}))
        ckpt = _fake_ckpt(names)
        out_dir = tmp_path / "models"
        result = lg.rebuild_live_graph(ckpt, out_dir)
        assert result == {"n_accounts": 0, "n_transactions": 0}
        graph = torch.load(out_dir / "graph.pt", map_location="cpu", weights_only=False)
        assert graph["account_ids"] == []

    def test_rebuilds_a_graph_matching_the_full_accumulated_live_population(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path / "live")
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}, {"src": "B", "dst": "C", "amount": 50}])

        tx_for_names = lg.read_live()
        accounts_for_names = lg._accounts_from_tx(tx_for_names)
        names = self._real_feature_names(accounts_for_names, tx_for_names)
        ckpt = _fake_ckpt(names)

        out_dir = tmp_path / "models"
        result = lg.rebuild_live_graph(ckpt, out_dir)

        assert result["n_accounts"] == 3   # A, B, C
        assert result["n_transactions"] == 2
        graph = torch.load(out_dir / "graph.pt", map_location="cpu", weights_only=False)
        assert set(graph["account_ids"]) == {"A", "B", "C"}
        assert graph["n_tx"] == 2
        assert graph["feature_names"] == names

    def test_mismatched_feature_names_raises_instead_of_silently_mis_scoring(self, monkeypatch, tmp_path):
        # if the live store's feature set no longer matches what the trained model expects (e.g. the model
        # was retrained with a different feature list since this live store was last rebuilt), this MUST
        # raise rather than silently feed misaligned columns into a model that will score them anyway
        _use_tmp_store(monkeypatch, tmp_path / "live")
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}])
        ckpt = _fake_ckpt(["totally_wrong_feature_set"])
        with pytest.raises(ValueError, match="feature_names"):
            lg.rebuild_live_graph(ckpt, tmp_path / "models")


class TestMain:
    def test_missing_checkpoint_exits_with_a_helpful_message(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr("sys.argv", ["livegraph.py", "--models", str(tmp_path / "nope")])
        with pytest.raises(SystemExit, match="not found"):
            lg.main()

    def test_reset_wipes_the_live_store_and_writes_a_blank_graph_without_touching_gnn_model(self, monkeypatch, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        ckpt = _fake_ckpt(["f1", "f2"])
        torch.save(ckpt, models_dir / "gnn_model.pt")
        before = (models_dir / "gnn_model.pt").read_bytes()

        _use_tmp_store(monkeypatch, tmp_path / "live")
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}])
        assert lg._live_tx_path().exists()

        monkeypatch.setattr("sys.argv", ["livegraph.py", "--models", str(models_dir), "--reset"])
        lg.main()

        assert not lg._live_tx_path().exists()
        assert (models_dir / "graph.pt").exists()
        graph = torch.load(models_dir / "graph.pt", map_location="cpu", weights_only=False)
        assert graph["account_ids"] == []
        assert (models_dir / "gnn_model.pt").read_bytes() == before  # the trained weights are untouched

    def test_without_reset_rebuilds_from_the_existing_live_store(self, monkeypatch, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        _use_tmp_store(monkeypatch, tmp_path / "live")
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}])
        tx = lg.read_live()
        accounts = lg._accounts_from_tx(tx)
        from gnn_aml_core.features import build_node_features
        _, names = build_node_features(accounts, tx)
        torch.save(_fake_ckpt(names), models_dir / "gnn_model.pt")

        monkeypatch.setattr("sys.argv", ["livegraph.py", "--models", str(models_dir)])
        lg.main()

        graph = torch.load(models_dir / "graph.pt", map_location="cpu", weights_only=False)
        assert graph["account_ids"] == ["A", "B"]


class TestRemoveAccounts:
    """The inverse of ingest() -- the confirmed design for the Intake tab's "clear temporary data" control:
    accounts disappear from the live graph, while alerts/decision history (a separate store this module has
    no concept of at all) are left completely alone by definition, since this function only ever touches
    transactions.csv."""

    def test_removing_an_account_drops_every_transaction_touching_it_as_either_side(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        lg.ingest([
            {"src": "A", "dst": "B", "amount": 100},   # touches B as dst
            {"src": "B", "dst": "C", "amount": 50},    # touches B as src
            {"src": "C", "dst": "D", "amount": 20},    # doesn't touch B at all
        ])
        remaining = lg.remove_accounts(["B"])
        assert len(remaining) == 1
        assert remaining.iloc[0]["src"] == "C"
        assert remaining.iloc[0]["dst"] == "D"

    def test_removing_several_accounts_at_once(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        lg.ingest([
            {"src": "A", "dst": "B", "amount": 100},
            {"src": "C", "dst": "D", "amount": 20},
            {"src": "E", "dst": "F", "amount": 5},
        ])
        remaining = lg.remove_accounts(["A", "B", "C", "D"])
        assert len(remaining) == 1
        assert set(remaining.iloc[0][["src", "dst"]]) == {"E", "F"}

    def test_removal_persists_to_disk(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}, {"src": "C", "dst": "D", "amount": 20}])
        lg.remove_accounts(["A"])
        assert len(lg.read_live()) == 1
        assert lg.read_live().iloc[0]["src"] == "C"

    def test_removing_an_account_that_was_never_ingested_is_a_no_op(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}])
        remaining = lg.remove_accounts(["NOPE"])
        assert len(remaining) == 1

    def test_removing_from_an_empty_store_is_a_no_op(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        remaining = lg.remove_accounts(["A"])
        assert len(remaining) == 0

    def test_empty_account_id_list_is_a_no_op(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}])
        remaining = lg.remove_accounts([])
        assert len(remaining) == 1

    def test_removing_everyone_leaves_an_empty_but_correctly_columned_store(self, monkeypatch, tmp_path):
        _use_tmp_store(monkeypatch, tmp_path)
        lg.ingest([{"src": "A", "dst": "B", "amount": 100}])
        remaining = lg.remove_accounts(["A", "B"])
        assert len(remaining) == 0
        assert list(remaining.columns) == lg.TX_COLUMNS
        # and it's really on disk, not just the in-memory return value
        assert len(lg.read_live()) == 0
