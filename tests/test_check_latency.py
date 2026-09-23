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


class TestWarmUp:
    def test_uses_accounts_outside_the_exclude_set(self):
        calls = []

        def fake_call(base, method, path, body=None, timeout=30):
            if path.startswith("/accounts/sample"):
                return {"suspicious": ["A", "B"], "benign": ["C", "D", "E"]}
            calls.append(body["account_ids"][0])
            return {"results": [{"account_id": body["account_ids"][0], "score": 0.5, "flagged": False, "cached": False}]}

        orig_call = _lat.call
        _lat.call = fake_call
        try:
            _lat.warm_up("http://fake", exclude={"A", "B", "C"}, n=5)
        finally:
            _lat.call = orig_call
        assert set(calls) == {"D", "E"}  # only the accounts NOT in exclude were used

    def test_never_uses_more_than_n_accounts(self):
        calls = []

        def fake_call(base, method, path, body=None, timeout=30):
            if path.startswith("/accounts/sample"):
                return {"suspicious": [f"S{i}" for i in range(10)], "benign": []}
            calls.append(body["account_ids"][0])
            return {"results": [{"account_id": body["account_ids"][0], "score": 0.5, "flagged": False, "cached": False}]}

        orig_call = _lat.call
        _lat.call = fake_call
        try:
            _lat.warm_up("http://fake", exclude=set(), n=3)
        finally:
            _lat.call = orig_call
        assert len(calls) == 3

    def test_does_nothing_if_sample_endpoint_is_unreachable(self):
        def failing_call(base, method, path, body=None, timeout=30):
            raise ConnectionError("simulated")

        orig_call = _lat.call
        _lat.call = failing_call
        try:
            _lat.warm_up("http://fake", exclude=set(), n=5)  # must not raise
        finally:
            _lat.call = orig_call

    def test_does_nothing_if_every_sampled_account_is_excluded(self):
        calls = []

        def fake_call(base, method, path, body=None, timeout=30):
            if path.startswith("/accounts/sample"):
                return {"suspicious": ["A", "B"], "benign": []}
            calls.append(body["account_ids"][0])
            return {}

        orig_call = _lat.call
        _lat.call = fake_call
        try:
            _lat.warm_up("http://fake", exclude={"A", "B"}, n=5)
        finally:
            _lat.call = orig_call
        assert calls == []


class TestStratifyByDegree:
    def test_returns_none_for_a_missing_graph_file(self, tmp_path):
        result = _lat.stratify_by_degree(tmp_path / "does_not_exist.pt", 10)
        assert result is None

    def test_identifies_a_low_degree_hub_adjacent_account_as_high_neighbourhood(self, tmp_path):
        # The exact scenario that made the OLD (own-degree) stratification mislead on real data: a "collector"
        # account (aml_synth's merchant/marketplace hard negative) with thousands of distinct customers. A
        # customer's own degree is tiny (they only ever paid the collector once), but the collector's entire
        # fan-in belongs to their 2-hop neighbourhood -- so they must be identified as HIGH neighbourhood size,
        # not low, despite their own degree being far smaller than an ordinary "fake hub" with many direct but
        # non-fanning neighbours. This test only needs torch (to build the .pt file) and scipy, not the GNN.
        torch = pytest.importorskip("torch")
        pytest.importorskip("scipy")

        n = 3000
        src, dst = [], []
        for i in range(10, 2010):        # 2,000 customers of one collector (node 0); each customer's only edge
            src += [i, 0]
            dst += [0, i]
        for i in range(2010, 2510):      # a "fake hub" (node 1) with 500 direct, non-fanning neighbours
            src += [1, i]
            dst += [i, 1]
        ei = torch.tensor([src, dst], dtype=torch.long)
        x = torch.zeros(n, 4)
        account_ids = [f"ACC{i}" for i in range(n)]
        graph = {"x": x, "edge_index": ei, "account_ids": account_ids, "n_tx": ei.shape[1] // 2}
        p = tmp_path / "graph.pt"
        torch.save(graph, p)

        buckets = _lat.stratify_by_degree(p, n_per_bucket=5, hops=2, seed=0)
        assert buckets is not None
        high_ids = set(buckets["high (large neighbourhood, likely hub-adjacent)"])
        # a customer of the collector (tiny own degree) must be classified as high-neighbourhood, not low
        assert "ACC15" in high_ids
        assert "ACC1" not in high_ids  # the fake hub's neighbourhood does not actually balloon

    def test_same_seed_gives_identical_bucket_selection(self, tmp_path):
        # Without this, argsort on a fixed graph is fully deterministic, so re-running the benchmark within
        # Redis's cache TTL would always pick the exact same accounts and find every one already cached,
        # making a fresh cold measurement impossible -- this is a real bug this test guards against regressing.
        torch = pytest.importorskip("torch")
        pytest.importorskip("scipy")
        p = self._build_random_graph(torch, tmp_path, n=500, seed=1)
        b1 = _lat.stratify_by_degree(p, n_per_bucket=5, hops=2, seed=42)
        b2 = _lat.stratify_by_degree(p, n_per_bucket=5, hops=2, seed=42)
        assert b1 == b2

    def test_different_seeds_usually_give_different_selections(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("scipy")
        p = self._build_random_graph(torch, tmp_path, n=500, seed=1)
        b1 = _lat.stratify_by_degree(p, n_per_bucket=5, hops=2, seed=1)
        b2 = _lat.stratify_by_degree(p, n_per_bucket=5, hops=2, seed=2)
        assert b1 != b2

    def test_no_seed_gives_different_selections_across_calls(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("scipy")
        p = self._build_random_graph(torch, tmp_path, n=500, seed=1)
        b1 = _lat.stratify_by_degree(p, n_per_bucket=5, hops=2, seed=None)
        b2 = _lat.stratify_by_degree(p, n_per_bucket=5, hops=2, seed=None)
        assert b1 != b2

    @staticmethod
    def _build_random_graph(torch, tmp_path, n: int, seed: int):
        import numpy as np
        rng = np.random.default_rng(seed)
        n_edges = n * 4
        src = rng.integers(0, n, n_edges)
        dst = rng.integers(0, n, n_edges)
        ei = torch.tensor(np.array([np.concatenate([src, dst]), np.concatenate([dst, src])]), dtype=torch.long)
        graph = {"x": torch.zeros(n, 4), "edge_index": ei, "account_ids": [f"ACC{i}" for i in range(n)], "n_tx": n_edges}
        p = tmp_path / f"graph_{seed}.pt"
        torch.save(graph, p)
        return p
