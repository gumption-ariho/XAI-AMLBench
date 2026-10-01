"""Tests for gnn_aml_core.synthaml, using a schema-exact synthetic stand-in (no real SynthAML data needed) --
same methodology as test_elliptic.py. Confirms the loader, feature engineering, and quarter-respecting split
logic work correctly against the documented schema (Jensen et al., 2023)."""
import numpy as np
import pandas as pd
import pytest

from gnn_aml_core.synthaml import (_check_quarter_boundary, build_alert_features, load_synthaml, quarter_split)


@pytest.fixture
def stub_dir(tmp_path):
    rng = np.random.default_rng(0)
    n_alerts = 200
    alert_ids = [f"AL{i:05d}" for i in range(n_alerts)]
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 730, n_alerts), unit="D")
    outcomes = (rng.random(n_alerts) < 0.17).astype(int)
    pd.DataFrame({"alert_id": alert_ids, "date": dates, "outcome": outcomes}).to_csv(
        tmp_path / "synthetic_alerts.csv", index=False)

    types, entries = ["card", "cash", "international", "wire"], ["credit", "debit"]
    rows = []
    for aid in alert_ids:
        for _ in range(rng.integers(5, 40)):
            rows.append({"alert_id": aid, "timestamp": int(rng.integers(0, 365 * 86400)),
                        "entry": rng.choice(entries), "type": rng.choice(types), "amount": float(rng.normal(0, 1))})
    pd.DataFrame(rows).to_csv(tmp_path / "synthetic_transactions.csv", index=False)
    return tmp_path


class TestLoadSynthaml:
    def test_correct_shape(self, stub_dir):
        arrays = load_synthaml(stub_dir)
        assert arrays["x"].shape == (200, 56)
        assert len(arrays["feature_names"]) == 56
        assert arrays["y"].shape == (200,)

    def test_features_are_not_all_the_missing_sentinel(self, stub_dir):
        arrays = load_synthaml(stub_dir)
        assert not (arrays["x"] == -3.0).all()

    def test_all_values_finite_no_nan(self, stub_dir):
        # regression guard: pandas' sample std is NaN for a group of exactly one transaction, which crashed
        # every downstream sklearn model until fixed -- this pins that it no longer happens
        arrays = load_synthaml(stub_dir)
        assert np.isfinite(arrays["x"]).all()

    def test_missing_files_gives_a_clear_actionable_error(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="figshare"):
            load_synthaml(empty_dir)


class TestBuildAlertFeatures:
    def test_empty_group_gets_zero_count_and_sentinel_for_everything_else(self):
        alerts = pd.DataFrame({"alert_id": ["A1"], "date": [pd.Timestamp("2020-06-01")], "outcome": [0]})
        tx = pd.DataFrame([{"alert_id": "A1", "type": "card", "entry": "credit", "amount": 1.0}])
        x, names, y, dates = build_alert_features(alerts, tx)
        assert x[0, names.index("wire_debit_count")] == 0.0
        assert x[0, names.index("wire_debit_min")] == -3.0
        assert x[0, names.index("card_credit_min")] == 1.0

    def test_single_transaction_group_gives_zero_std_not_nan(self):
        alerts = pd.DataFrame({"alert_id": ["A1"], "date": [pd.Timestamp("2020-06-01")], "outcome": [0]})
        tx = pd.DataFrame([{"alert_id": "A1", "type": "card", "entry": "credit", "amount": 5.0}])
        x, names, y, dates = build_alert_features(alerts, tx)
        std_idx = names.index("card_credit_std")
        assert x[0, std_idx] == 0.0
        assert np.isfinite(x).all()

    def test_outcome_correctly_becomes_y(self):
        alerts = pd.DataFrame({"alert_id": ["A1", "A2"], "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                              "outcome": [1, 0]})
        tx = pd.DataFrame([{"alert_id": "A1", "type": "card", "entry": "credit", "amount": 1.0}])
        x, names, y, dates = build_alert_features(alerts, tx)
        assert list(y) == [1.0, 0.0]


class TestQuarterBoundary:
    def test_valid_quarter_boundaries_accepted(self):
        for d in ("2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01"):
            _check_quarter_boundary(d)  # must not raise

    def test_non_quarter_date_rejected_with_clear_message(self):
        with pytest.raises(ValueError, match="quarter boundary"):
            _check_quarter_boundary("2021-02-15")


class TestQuarterSplit:
    def test_no_overlap_between_splits(self, stub_dir):
        arrays = load_synthaml(stub_dir)
        tr, va, te = quarter_split(arrays, "2021-01-01", "2021-01-01", "2022-01-01")
        assert len(tr) > 0 and len(va) > 0 and len(te) > 0
        assert set(tr).isdisjoint(va)
        assert set(tr).isdisjoint(te)
        assert set(va).isdisjoint(te)

    def test_rejects_a_non_quarter_split_date(self, stub_dir):
        arrays = load_synthaml(stub_dir)
        with pytest.raises(ValueError, match="quarter boundary"):
            quarter_split(arrays, "2021-02-15", "2021-01-01", "2022-01-01")

    def test_test_split_only_contains_alerts_in_the_requested_window(self, stub_dir):
        arrays = load_synthaml(stub_dir)
        tr, va, te = quarter_split(arrays, "2021-01-01", "2021-01-01", "2022-01-01")
        test_dates = pd.to_datetime(arrays["dates"][te])
        assert (test_dates >= pd.Timestamp("2021-01-01")).all()
        assert (test_dates < pd.Timestamp("2022-01-01")).all()
