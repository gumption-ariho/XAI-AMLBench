"""Tests for gnn_aml_core.company_fraud. Fixtures are hand-crafted to match the real dataset's schema exactly
as confirmed by direct inspection of the real files (see the module's own docstring), not assumed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnn_aml_core.company_fraud import build_window_features, load_split


def _write_split(tmp_path: Path, split: str) -> None:
    (tmp_path / f"companies_{split}.csv").write_text(
        ",0,1,2,3,4,5\ncompany_A,0.1,0.2,0.3,0.4,0.5,0.6\ncompany_B,1.1,1.2,1.3,1.4,1.5,1.6\n"
    )
    (tmp_path / f"transactions_{split}.csv").write_text(
        "transactionId,0,1,2\ntransaction_1,10.0,20.0,30.0\ntransaction_2,40.0,50.0,60.0\n"
        "transaction_3,70.0,80.0,90.0\ntransaction_4,5.0,5.0,5.0\n"
    )
    (tmp_path / f"event_order_{split}.csv").write_text(
        "transactionId,eventAt\ntransaction_1,1\ntransaction_2,2\ntransaction_3,3\ntransaction_4,1\n"
    )
    # many-to-many: transaction_2 deliberately belongs to BOTH company_A_window_1 and company_A_window_2,
    # mirroring the real data's confirmed many-to-many transaction<->window mapping
    (tmp_path / f"time_series_ids_{split}.csv").write_text(
        "transactionId,time_series_ids\ntransaction_1,company_A_window_1\ntransaction_2,company_A_window_1\n"
        "transaction_2,company_A_window_2\ntransaction_3,company_A_window_2\ntransaction_4,company_B_window_1\n"
    )
    (tmp_path / f"fraud_labels_{split}.csv").write_text(
        ",isFraudUser\ncompany_A_window_1,True\ncompany_A_window_2,False\n"
        "company_B_window_1,False\ncompany_UNKNOWN_window_1,True\n"
    )


@pytest.fixture
def split(tmp_path: Path) -> dict[str, pd.DataFrame]:
    _write_split(tmp_path, "train")
    return load_split(tmp_path, "train")


class TestLoadSplit:
    def test_loads_all_five_files_with_expected_shapes(self, split):
        assert split["companies"].shape == (2, 6)
        assert split["transactions"].shape == (4, 4)   # transactionId is a real column here (unlike companies
                                                        # and fraud_labels, which use index_col=0), so it is
                                                        # counted: 3 numbered feature columns + transactionId
        assert split["time_series_ids"].shape == (5, 2)
        assert split["fraud_labels"].shape == (4, 1)

    def test_companies_and_fraud_labels_are_indexed_by_their_key_column(self, split):
        assert list(split["companies"].index) == ["company_A", "company_B"]
        assert "company_A_window_1" in split["fraud_labels"].index


class TestBuildWindowFeatures:
    def test_row_count_matches_labelled_windows_not_transactions(self, split):
        x, y, names = build_window_features(split)
        assert x.shape[0] == 4   # 4 labelled windows, not 4 transactions (coincidence in this fixture) or 5 mappings

    def test_row_order_matches_fraud_labels_index(self, split):
        _, y, _ = build_window_features(split)
        assert y.tolist() == [1, 0, 0, 1]   # True, False, False, True in fraud_labels' own order

    def test_no_nan_or_inf_anywhere_in_output(self, split):
        x, _, _ = build_window_features(split)
        assert not np.isnan(x).any()
        assert not np.isinf(x).any()

    def test_a_transaction_shared_across_two_windows_contributes_to_both(self, split):
        # transaction_2 (col "0" = 40.0) belongs to BOTH company_A_window_1 and company_A_window_2 -- this is a
        # real, confirmed many-to-many mapping in the source data, not an assumption
        x, _, names = build_window_features(split)
        i_mean = names.index("0_mean")
        assert abs(x[0, i_mean] - 25.0) < 1e-5   # window_1: txns 1,2 -> (10+40)/2
        assert abs(x[1, i_mean] - 55.0) < 1e-5   # window_2: txns 2,3 -> (40+70)/2

    def test_a_singleton_window_gets_zero_std_not_nan(self, split):
        # a real bug this project already hit once before, in gnn_aml_core.synthaml: pandas' std of a single
        # value is NaN, not 0.0 -- proactively fixed here rather than rediscovered
        x, _, names = build_window_features(split)
        i_std = names.index("0_std")
        row = 2   # company_B_window_1 has exactly one transaction
        assert x[row, i_std] == 0.0

    def test_a_window_whose_company_has_no_match_degrades_to_zero_not_a_crash(self, split):
        x, _, names = build_window_features(split)
        row = 3   # company_UNKNOWN_window_1: no transactions AND no companies.csv match
        assert x[row].sum() == 0.0

    def test_company_features_are_correctly_attached_to_every_one_of_that_companys_windows(self, split):
        x, _, names = build_window_features(split)
        i_company0 = names.index("company_0")
        assert abs(x[0, i_company0] - 0.1) < 1e-6   # company_A_window_1 -> company_A's own feature "0"
        assert abs(x[1, i_company0] - 0.1) < 1e-6   # company_A_window_2 -> the SAME company_A feature
        assert abs(x[2, i_company0] - 1.1) < 1e-6   # company_B_window_1 -> company_B's own feature
