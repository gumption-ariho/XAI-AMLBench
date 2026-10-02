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
