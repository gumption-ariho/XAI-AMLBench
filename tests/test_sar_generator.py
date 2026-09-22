"""Tests for xai_explainer.sar_generator: the deterministic explanation -> facts -> narrative -> validation
pipeline. No PyTorch needed, so these always run."""
import re

import pytest

from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig, TYPOLOGIES
from xai_explainer import sar_generator as sar


def explanation_for(graph, typology: str) -> dict:
    """Build a synthetic /explain-shaped dict from one injected pattern of the given typology, the same way
    check_brief_v1.py section E does it, so build_facts sees realistic input."""
    A = graph.accounts.set_index("account_id")
    T = graph.transactions
    pid = sorted(T[T.typology == typology].pattern_id.unique())[0]
    sub = T[T.pattern_id == pid]
    subject = sub.src.value_counts().add(sub.dst.value_counts(), fill_value=0).idxmax()
    ids = list(dict.fromkeys([subject] + [x for r in sub.itertuples() for x in (r.src, r.dst)]))
    return {
        "account_id": subject, "risk_score": 0.92, "reporting_threshold": 10_000,
        "nodes": [{"account_id": i, "account_type": A.loc[i, "account_type"], "country": A.loc[i, "country"]} for i in ids],
        "edges": [{"tx_id": r.tx_id, "src": r.src, "dst": r.dst, "amount": float(r.amount), "timestamp": int(r.timestamp),
                   "cross_border": int(r.cross_border)} for r in sub.itertuples()],
        "top_features": [{"feature": "near_thr_ratio", "weight": 0.4}, {"feature": "burst_6h", "weight": 0.2}],
    }


@pytest.fixture(scope="module")
def pattern_graph():
    return AMLGraphGenerator(GeneratorConfig(n_accounts=1200, n_background_tx=5000, n_patterns_per_typology=4, seed=3)).generate()


@pytest.fixture(params=TYPOLOGIES)
def typology(request):
    return request.param


@pytest.fixture
def explanation(typology, pattern_graph):
    return explanation_for(pattern_graph, typology)


class TestBuildFacts:
    def test_facts_contain_required_keys(self, explanation):
        f = sar.build_facts(explanation)
        for key in ("subject", "typology", "score_pct", "n_accounts", "n_tx", "total", "start", "end"):
            assert key in f

    def test_typology_correctly_inferred(self, explanation, typology):
        # explanation_for() builds the subgraph from a pattern the generator itself labelled with this typology,
        # so infer_typology's structural heuristics should recover the same label. This caught a real bug: the
        # heuristics were written against an earlier version of the generator and silently drifted out of sync
        # after later changes to typical pattern size and timing.
        f = sar.build_facts(explanation)
        assert f["typology"] == typology

    def test_no_edges_gives_unclassified(self):
        exp = {"account_id": "ACC0000001", "risk_score": 0.5, "reporting_threshold": 10_000,
               "nodes": [{"account_id": "ACC0000001", "account_type": "individual", "country": "US"}], "edges": []}
        f = sar.build_facts(exp)
        assert f["typology"] == "unclassified"

    def test_score_pct_rounds_correctly(self, explanation):
        explanation["risk_score"] = 0.876
        f = sar.build_facts(explanation)
        assert f["score_pct"] == "88%"

    def test_total_matches_sum_of_edge_amounts(self, explanation):
        f = sar.build_facts(explanation)
        expected = sum(e["amount"] for e in explanation["edges"])
        assert f["total"] == f"${expected:,.0f}"


class TestTemplateNarrative:
    def test_is_deterministic(self, explanation):
        f = sar.build_facts(explanation)
        a = sar.template_narrative(f)
        b = sar.template_narrative(f)
        assert a == b

    def test_has_three_to_four_sentences(self, explanation):
        f = sar.build_facts(explanation)
        text = sar.template_narrative(f)
        n = len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if s.strip()])
        assert 3 <= n <= 4

    def test_contains_date_amount_and_account_id(self, explanation):
        f = sar.build_facts(explanation)
        text = sar.template_narrative(f)
        assert re.search(r"\d{4}-\d{2}-\d{2}", text)
        assert re.search(r"\$[\d,]+", text)
        assert re.search(r"ACC\d+", text)

    def test_no_banned_words(self, explanation):
        f = sar.build_facts(explanation)
        text = sar.template_narrative(f).lower()
        for w in sar._BANNED:
            assert w not in text

    def test_validator_accepts_the_template_output(self, explanation):
        f = sar.build_facts(explanation)
        sheet = sar.render_fact_sheet(f)
        text = sar.template_narrative(f)
        ok, problems = sar.validate_narrative(text, sheet)
        assert ok, problems


class TestValidateNarrative:
    def test_rejects_a_number_not_in_the_fact_sheet(self):
        sheet = "Account ACC0000001 sent $5,000 to ACC0000002."
        bad = "Account ACC0000001 sent $999,999 to ACC0000002."
        ok, problems = sar.validate_narrative(bad, sheet)
        assert not ok
        assert problems

    def test_accepts_exact_reuse_of_fact_sheet_numbers(self):
        # validate_narrative also requires 3-4 sentences, so this checks a realistic narrative shape, not just
        # the number-reuse rule in isolation.
        sheet = "Account ACC0000001 sent $5,000 to ACC0000002 on 2026-01-05, in 3 transactions."
        good = ("Account ACC0000001 sent $5,000 to ACC0000002 on 2026-01-05. The transfer involved 3 transactions. "
                "This activity was flagged for review.")
        ok, problems = sar.validate_narrative(good, sheet)
        assert ok, problems

    def test_rejects_an_invented_account_id(self):
        sheet = "Account ACC0000001 sent $5,000 to ACC0000002."
        bad = "Account ACC0000001 sent $5,000 to ACC9999999."
        ok, _ = sar.validate_narrative(bad, sheet)
        assert not ok


class TestNarrativeHash:
    def test_same_text_same_hash(self):
        assert sar.narrative_hash("hello world") == sar.narrative_hash("hello world")

    def test_different_text_different_hash(self):
        assert sar.narrative_hash("hello world") != sar.narrative_hash("goodbye world")

    def test_hash_is_a_hex_string(self):
        h = sar.narrative_hash("some narrative text")
        assert re.fullmatch(r"[0-9a-f]+", h)


class TestInferTypology:
    def test_empty_edges_is_unclassified(self):
        assert sar.infer_typology("ACC1", [], [], 10_000) == "unclassified"

    def test_two_shell_accounts_infers_shell_company(self):
        nodes = [{"account_id": "A", "account_type": "shell", "country": "VG"},
                 {"account_id": "B", "account_type": "shell", "country": "KY"},
                 {"account_id": "C", "account_type": "business", "country": "US"}]
        edges = [{"src": "C", "dst": "A", "amount": 5000, "timestamp": 0, "cross_border": 1},
                 {"src": "A", "dst": "B", "amount": 4900, "timestamp": 100, "cross_border": 1}]
        assert sar.infer_typology("C", nodes, edges, 10_000) == "shell_company"
