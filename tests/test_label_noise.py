"""Tests for aml_synth.label_noise: realistic investigator-error simulation layered on top of perfect
synthetic ground truth. Pure pandas/numpy, no torch needed."""
import pandas as pd
import pytest

from aml_synth.label_noise import inject_label_noise, label_noise_report


@pytest.fixture
def accounts():
    return pd.DataFrame({"account_id": [f"A{i}" for i in range(1000)], "is_suspicious": [1] * 100 + [0] * 900})


class TestInjectLabelNoise:
    def test_adds_noisy_column_without_removing_ground_truth(self, accounts):
        out = inject_label_noise(accounts, false_negative_rate=0.2, false_positive_rate=0.05, seed=1)
        assert "is_suspicious_noisy" in out.columns
        assert "is_suspicious" in out.columns

    def test_zero_rates_produce_no_noise_at_all(self, accounts):
        out = inject_label_noise(accounts, false_negative_rate=0.0, false_positive_rate=0.0, seed=1)
        assert (out["is_suspicious_noisy"] == out["is_suspicious"]).all()

    def test_rate_of_one_flips_every_suspicious_account(self, accounts):
        out = inject_label_noise(accounts, false_negative_rate=1.0, false_positive_rate=0.0, seed=1)
        assert (out.loc[out["is_suspicious"] == 1, "is_suspicious_noisy"] == 0).all()

    def test_rate_of_one_flips_every_benign_account(self, accounts):
        out = inject_label_noise(accounts, false_negative_rate=0.0, false_positive_rate=1.0, seed=1)
        assert (out.loc[out["is_suspicious"] == 0, "is_suspicious_noisy"] == 1).all()

    def test_realised_rate_is_close_to_requested(self, accounts):
        out = inject_label_noise(accounts, false_negative_rate=0.2, false_positive_rate=0.05, seed=1)
        rep = label_noise_report(out)
        assert rep["false_negative_rate_realised"] == pytest.approx(0.2, abs=0.08)
        assert rep["false_positive_rate_realised"] == pytest.approx(0.05, abs=0.04)

    def test_negative_rate_raises_a_clear_error(self, accounts):
        with pytest.raises(ValueError, match="percentage"):
            inject_label_noise(accounts, false_negative_rate=-0.1)

    def test_rate_above_one_raises_a_clear_error(self, accounts):
        with pytest.raises(ValueError, match="percentage"):
            inject_label_noise(accounts, false_negative_rate=15)  # a percentage passed by mistake

    def test_original_dataframe_is_never_mutated(self, accounts):
        original = accounts.copy()
        inject_label_noise(accounts, false_negative_rate=0.5, seed=1)
        assert (accounts["is_suspicious"] == original["is_suspicious"]).all()
        assert "is_suspicious_noisy" not in accounts.columns

    def test_same_seed_is_reproducible(self, accounts):
        out1 = inject_label_noise(accounts, false_negative_rate=0.3, seed=42)
        out2 = inject_label_noise(accounts, false_negative_rate=0.3, seed=42)
        assert (out1["is_suspicious_noisy"] == out2["is_suspicious_noisy"]).all()

    def test_different_seeds_usually_give_different_noise(self, accounts):
        out1 = inject_label_noise(accounts, false_negative_rate=0.3, seed=1)
        out2 = inject_label_noise(accounts, false_negative_rate=0.3, seed=2)
        assert not (out1["is_suspicious_noisy"] == out2["is_suspicious_noisy"]).all()


class TestLabelNoiseReport:
    def test_raises_a_clear_error_without_prior_noise_injection(self, accounts):
        with pytest.raises(KeyError, match="inject_label_noise"):
            label_noise_report(accounts)

    def test_reports_the_correct_account_counts(self, accounts):
        out = inject_label_noise(accounts, false_negative_rate=0.2, seed=1)
        rep = label_noise_report(out)
        assert rep["n_accounts"] == 1000
        assert rep["n_truly_suspicious"] == 100
        assert rep["n_truly_benign"] == 900
