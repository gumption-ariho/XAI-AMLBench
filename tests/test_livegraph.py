"""Tests for gnn_aml_core.livegraph's live (non-synthetic) transaction store -- ingest()/read_live(), and
remove_accounts() (the backing function for POST /admin/remove-live / backend's POST /intake/clear, the Intake
tab's "clear temporary data" control). rebuild_live_graph()/write_blank_graph() need PyTorch and the trained
model's checkpoint shape, so they're out of scope here (and this module's own existing test suite, if any,
should cover them under the `torch` marker); this file only exercises the on-disk CSV manipulation, which is
exactly what remove_accounts() is responsible for getting right.
"""
import os

os.environ.setdefault("LIVE_DATA_DIR", "data/live")   # read once at import time by the module under test

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
