"""Tests for check_latency_v1.py's pure-Python statistics helpers (percentile, summarize). No network, no
PyTorch, no external services -- these always run. The script itself is a standalone diagnostic tool (not part
of the installable package), loaded here by file path since it lives at the project root, not in a package."""
import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("check_latency_v1", Path(__file__).resolve().parent.parent / "check_latency_v1.py")
_lat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_lat)


class TestPercentile:
    def test_median_of_evenly_spaced_values(self):
        assert _lat.percentile(list(range(1, 101)), 0.5) == pytest.approx(50.5, abs=1.0)

    def test_p0_is_the_minimum(self):
        assert _lat.percentile([5, 1, 9, 3], 0.0) == 1

    def test_p100_is_the_maximum(self):
        assert _lat.percentile([5, 1, 9, 3], 1.0) == 9

    def test_empty_list_is_nan(self):
        assert math.isnan(_lat.percentile([], 0.5))

    def test_single_value_returns_that_value_at_any_percentile(self):
        assert _lat.percentile([42], 0.5) == 42
        assert _lat.percentile([42], 0.99) == 42

    def test_constant_values_return_that_constant(self):
        assert _lat.percentile([10.0] * 50, 0.95) == 10.0

    def test_unsorted_input_is_handled_correctly(self):
        assert _lat.percentile([3, 1, 2], 0.5) == 2


class TestSummarize:
    def test_basic_statistics(self):
        s = _lat.summarize("label", [10, 20, 30, 40, 100])
        assert s["n"] == 5
        assert s["mean"] == pytest.approx(40.0)
        assert s["max"] == 100

    def test_empty_list_reports_zero_n_and_nothing_else(self):
        s = _lat.summarize("label", [])
        assert s["n"] == 0
        assert "mean" not in s

    def test_meets_target_true_when_fast(self):
        s = _lat.summarize("label", [50.0] * 20)
        assert s["meets_target_p95"] is True

    def test_meets_target_false_when_slow(self):
        s = _lat.summarize("label", [500.0] * 20)
        assert s["meets_target_p95"] is False

    def test_label_is_preserved(self):
        assert _lat.summarize("my bucket", [1.0])["label"] == "my bucket"


class TestStratifyByDegree:
    def test_returns_none_for_a_missing_graph_file(self, tmp_path):
        result = _lat.stratify_by_degree(tmp_path / "does_not_exist.pt", 10)
        assert result is None
