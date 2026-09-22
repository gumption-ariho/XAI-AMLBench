"""Tests for aml_synth.graph_generator, checked against the brief's own Module 1 acceptance criteria (see
check_brief_v1.py section B for the same checks run as a standalone scorecard)."""
import networkx as nx
import numpy as np
import pytest

from aml_synth.graph_generator import AMLGraphGenerator, DAY, GeneratorConfig, START_TS, TYPOLOGIES


def make(**kw):
    cfg = GeneratorConfig(n_accounts=kw.pop("n_accounts", 1500), n_background_tx=kw.pop("n_background_tx", 6000),
                          n_patterns_per_typology=kw.pop("n_patterns_per_typology", 5), **kw)
    return AMLGraphGenerator(cfg).generate()


class TestBriefAcceptanceCriteria:
    """Brief 1.1: "Testing Acceptance Criteria" section, one test per bullet."""

    def test_node_count_within_10_percent(self):
        n = 2000
        g = make(n_accounts=n, seed=1)
        assert abs(len(g.accounts) - n) / n <= 0.10

    def test_valid_directed_multigraph(self):
        g = make(seed=2)
        G = g.to_networkx()
        assert G.is_directed() and G.is_multigraph()
        assert G.number_of_nodes() == len(g.accounts)
        assert G.number_of_edges() == len(g.transactions)

    @pytest.mark.parametrize("seed", range(8))
    def test_timestamps_inside_requested_window(self, seed):
        days = 30
        g = make(days=days, n_accounts=1200, n_background_tx=5000, n_patterns_per_typology=8, seed=seed)
        t = g.transactions.timestamp
        assert t.min() >= START_TS
        assert t.max() <= START_TS + days * DAY

    def test_median_amount_in_range(self):
        g = make(seed=3)
        assert 500 <= g.transactions.amount.median() <= 2000

    def test_all_five_typologies_present(self):
        g = make(seed=4)
        present = set(g.transactions.loc[g.transactions.is_laundering == 1, "typology"])
        assert set(TYPOLOGIES) <= present

    def test_smurfing_reaches_required_deposit_count(self):
        g = make(n_accounts=5000, n_background_tx=20000, n_patterns_per_typology=2, seed=5)
        sizes = g.transactions[g.transactions.typology == "smurfing"].groupby("pattern_id").size()
        assert sizes.max() >= 1000


class TestGeneratorProperties:
    def test_reproducible_with_same_seed(self):
        a = make(seed=42).transactions
        b = make(seed=42).transactions
        assert a.equals(b)

    def test_different_seeds_differ(self):
        a = make(seed=1).transactions
        b = make(seed=2).transactions
        assert not a.equals(b)

    def test_no_self_loops_in_background_traffic(self, small_graph):
        assert not (small_graph.transactions.src == small_graph.transactions.dst).any()

    def test_no_missing_values(self, small_graph):
        assert not small_graph.accounts.isna().any().any()
        assert not small_graph.transactions.isna().any().any()

    def test_every_transaction_endpoint_is_a_known_account(self, small_graph):
        ids = set(small_graph.accounts.account_id)
        assert set(small_graph.transactions.src) <= ids
        assert set(small_graph.transactions.dst) <= ids

    def test_account_ids_unique(self, small_graph):
        assert small_graph.accounts.account_id.is_unique

    def test_transaction_ids_unique(self, small_graph):
        assert small_graph.transactions.tx_id.is_unique

    def test_amounts_positive(self, small_graph):
        assert (small_graph.transactions.amount > 0).all()

    def test_cross_border_flag_matches_countries(self, small_graph):
        A = small_graph.accounts.set_index("account_id")
        T = small_graph.transactions
        expected = (A.loc[T.src, "country"].values != A.loc[T.dst, "country"].values)
        assert (T.cross_border.values.astype(bool) == expected).all()

    def test_suspicious_flag_matches_typology(self, small_graph):
        A = small_graph.accounts
        assert set(A[A.is_suspicious == 1].typology) <= set(TYPOLOGIES)
        assert set(A[A.is_suspicious == 0].typology) == {"none"}

    def test_risk_score_in_unit_interval(self, small_graph):
        assert small_graph.accounts.risk_score.between(0, 1).all()

    def test_summary_matches_dataframes(self, small_graph):
        s = small_graph.summary()
        assert s["accounts"] == len(small_graph.accounts)
        assert s["transactions"] == len(small_graph.transactions)
        assert s["suspicious_accounts"] == int(small_graph.accounts.is_suspicious.sum())

    def test_tiny_graph_does_not_crash(self):
        g = make(n_accounts=100, n_background_tx=200, n_patterns_per_typology=1, seed=9)
        assert len(g.accounts) > 0 and len(g.transactions) > 0

    def test_zero_patterns_still_produces_background_traffic(self):
        g = make(n_accounts=300, n_background_tx=500, n_patterns_per_typology=0, seed=10)
        assert len(g.transactions) > 0
        assert (g.accounts.is_suspicious == 0).all()

    def test_scales_to_larger_graph(self):
        g = make(n_accounts=8000, n_background_tx=30000, n_patterns_per_typology=6, seed=11)
        assert 7000 <= len(g.accounts) <= 9000
