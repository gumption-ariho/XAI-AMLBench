"""Tests for aml_synth.exporters. CSV/JSON/Cypher need no optional dependency; Parquet needs pyarrow and
Kafka/Neo4j need kafka-python/neo4j -- those are skipped (not failed) when the package is not installed, via
pytest.importorskip, the same pattern used for the PyTorch-dependent tests elsewhere in this suite."""
import json
import re

import pandas as pd
import pytest

from aml_synth import exporters


class TestExportCsv:
    def test_writes_both_files(self, small_graph, tmp_path):
        exporters.export_csv(small_graph, tmp_path)
        assert (tmp_path / "accounts.csv").exists()
        assert (tmp_path / "transactions.csv").exists()

    def test_row_counts_match_the_graph(self, small_graph, tmp_path):
        exporters.export_csv(small_graph, tmp_path)
        assert len(pd.read_csv(tmp_path / "accounts.csv")) == len(small_graph.accounts)
        assert len(pd.read_csv(tmp_path / "transactions.csv")) == len(small_graph.transactions)

    def test_columns_preserved(self, small_graph, tmp_path):
        exporters.export_csv(small_graph, tmp_path)
        acc = pd.read_csv(tmp_path / "accounts.csv")
        assert set(acc.columns) == set(small_graph.accounts.columns)

    def test_creates_the_output_directory_if_missing(self, small_graph, tmp_path):
        out = tmp_path / "nested" / "does_not_exist_yet"
        exporters.export_csv(small_graph, out)
        assert (out / "accounts.csv").exists()


class TestExportParquet:
    def test_writes_both_files_and_round_trips(self, small_graph, tmp_path):
        pytest.importorskip("pyarrow")
        exporters.export_parquet(small_graph, tmp_path)
        acc = pd.read_parquet(tmp_path / "accounts.parquet")
        tx = pd.read_parquet(tmp_path / "transactions.parquet")
        assert len(acc) == len(small_graph.accounts)
        assert len(tx) == len(small_graph.transactions)

    def test_preserves_integer_dtype_unlike_csv(self, small_graph, tmp_path):
        # Parquet is the "primary format for distributed processing" precisely because it keeps dtypes exactly;
        # this is the property CSV cannot offer, so it is worth asserting explicitly.
        pytest.importorskip("pyarrow")
        exporters.export_parquet(small_graph, tmp_path)
        tx = pd.read_parquet(tmp_path / "transactions.parquet")
        assert pd.api.types.is_integer_dtype(tx["timestamp"])

    def test_raises_a_clear_error_without_pyarrow(self, small_graph, tmp_path, monkeypatch):
        # Simulate pyarrow being absent regardless of what is actually installed here, so this test is meaningful
        # on a machine that DOES have pyarrow too.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "pyarrow":
                raise ImportError("simulated: pyarrow not installed")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="pyarrow"):
            exporters.export_parquet(small_graph, tmp_path)


class TestExportJson:
    def test_writes_graph_summary(self, small_graph, tmp_path):
        exporters.export_json(small_graph, tmp_path)
        assert (tmp_path / "graph_summary.json").exists()

    def test_is_valid_json_with_expected_top_level_keys(self, small_graph, tmp_path):
        exporters.export_json(small_graph, tmp_path)
        doc = json.loads((tmp_path / "graph_summary.json").read_text())
        for key in ("name", "synthetic", "contains_personal_data", "summary", "schema"):
            assert key in doc

    def test_marks_data_as_synthetic_with_no_personal_data(self, small_graph, tmp_path):
        exporters.export_json(small_graph, tmp_path)
        doc = json.loads((tmp_path / "graph_summary.json").read_text())
        assert doc["synthetic"] is True
        assert doc["contains_personal_data"] is False

    def test_summary_matches_graph_summary_method(self, small_graph, tmp_path):
        exporters.export_json(small_graph, tmp_path)
        doc = json.loads((tmp_path / "graph_summary.json").read_text())
        assert doc["summary"]["accounts"] == small_graph.summary()["accounts"]

    def test_schema_documents_every_real_column(self, small_graph, tmp_path):
        exporters.export_json(small_graph, tmp_path)
        doc = json.loads((tmp_path / "graph_summary.json").read_text())
        assert set(doc["schema"]["accounts"]) == set(small_graph.accounts.columns)
        assert set(doc["schema"]["transactions"]) == set(small_graph.transactions.columns)


class TestCypherHelpers:
    def test_lit_quotes_strings(self):
        assert exporters._lit("ACC0000001") == '"ACC0000001"'

    def test_lit_leaves_integers_bare(self):
        assert exporters._lit(42) == "42"

    def test_lit_formats_floats(self):
        assert exporters._lit(1234.5) == "1234.5"

    def test_lit_bool_lowercase(self):
        assert exporters._lit(True) == "true"
        assert exporters._lit(False) == "false"

    def test_lit_escapes_special_characters_safely(self):
        # a string containing a double quote must not break out of the Cypher string literal
        out = exporters._lit('say "hi"')
        assert out.startswith('"') and out.endswith('"')
        json.loads(out)  # Cypher accepts JSON-style strings; this should also parse as valid JSON

    def test_map_builds_a_cypher_map_literal(self):
        row = {"a": 1, "b": "x", "c": None}
        out = exporters._map(row, ["a", "b"])
        assert out == '{a: 1, b: "x"}'


class TestExportCypher:
    def test_writes_import_file(self, small_graph, tmp_path):
        exporters.export_cypher(small_graph, tmp_path)
        assert (tmp_path / "import.cypher").exists()

    def test_contains_a_uniqueness_constraint(self, small_graph, tmp_path):
        exporters.export_cypher(small_graph, tmp_path)
        text = (tmp_path / "import.cypher").read_text()
        assert "CREATE CONSTRAINT" in text and "account_id" in text

    def test_every_statement_ends_with_semicolon_and_balanced_brackets(self, small_graph, tmp_path):
        exporters.export_cypher(small_graph, tmp_path)
        text = (tmp_path / "import.cypher").read_text()
        statements = [s for s in text.split(";\n") if s.strip() and not s.strip().startswith("//")]
        for s in statements:
            assert s.count("{") == s.count("}")
            assert s.count("[") == s.count("]")
            assert s.count("(") == s.count(")")

    def test_row_counts_match_csv_export(self, small_graph, tmp_path):
        # Count only quoted-string data literals (account_id: "ACC...", tx_id: "TX...") produced once per real
        # row -- NOT the "row.account_id" / "row.tx_id" template references that also appear, once per batch, in
        # the UNWIND query text itself (a naive "account_id:" substring count double-counts those).
        exporters.export_csv(small_graph, tmp_path)
        exporters.export_cypher(small_graph, tmp_path)
        acc = pd.read_csv(tmp_path / "accounts.csv")
        tx = pd.read_csv(tmp_path / "transactions.csv")
        text = (tmp_path / "import.cypher").read_text()
        n_account_maps = len(re.findall(r'account_id:\s*"ACC', text))
        n_tx_maps = len(re.findall(r'tx_id:\s*"TX', text))
        assert n_account_maps == len(acc)
        assert n_tx_maps == len(tx)

    def test_batching_produces_multiple_unwind_blocks_for_a_large_graph(self, small_graph, tmp_path):
        exporters.export_cypher(small_graph, tmp_path, batch=50)
        text = (tmp_path / "import.cypher").read_text()
        n_unwind = text.count("UNWIND [")
        expected_min = len(small_graph.accounts) // 50  # at least this many batches, could be +1
        assert n_unwind >= expected_min


def _no_brokers_exception(kafka):
    """Mirrors the fallback in exporters.export_kafka: kafka-python has renamed this exception across
    versions, so tests must resolve it the same defensive way rather than assuming one fixed name."""
    errors = kafka.errors
    return (getattr(errors, "NoBrokersAvailable", None)
            or getattr(errors, "NoBrokersAvailableError", None)
            or getattr(errors, "KafkaError"))


class TestExportKafka:
    def test_publishes_every_transaction_with_correct_key(self, small_graph, monkeypatch):
        kafka = pytest.importorskip("kafka")
        sent = []

        class FakeProducer:
            def __init__(self, **kw):
                self.kw = kw

            def send(self, topic, key, value):
                sent.append((topic, key, value))

            def flush(self):
                pass

        monkeypatch.setattr(kafka, "KafkaProducer", FakeProducer)
        exporters.export_kafka(small_graph, bootstrap="fake:9092", topic="test.topic")
        assert len(sent) == len(small_graph.transactions)
        assert all(topic == "test.topic" for topic, _, _ in sent)
        assert {k for _, k, _ in sent} == set(small_graph.transactions.src)

    def test_retries_when_no_broker_available_then_succeeds(self, small_graph, monkeypatch):
        kafka = pytest.importorskip("kafka")
        NoBrokersAvailable = _no_brokers_exception(kafka)
        attempts = {"n": 0}

        class FlakyThenOkProducer:
            def __init__(self, **kw):
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise NoBrokersAvailable()

            def send(self, *a, **kw):
                pass

            def flush(self):
                pass

        monkeypatch.setattr(kafka, "KafkaProducer", FlakyThenOkProducer)
        monkeypatch.setattr(exporters.time, "sleep", lambda *_: None)  # don't actually wait in the test
        exporters.export_kafka(small_graph, bootstrap="fake:9092")
        assert attempts["n"] == 3

    def test_gives_up_after_repeated_failure(self, small_graph, monkeypatch):
        kafka = pytest.importorskip("kafka")
        NoBrokersAvailable = _no_brokers_exception(kafka)

        class AlwaysFailsProducer:
            def __init__(self, **kw):
                raise NoBrokersAvailable()

        monkeypatch.setattr(kafka, "KafkaProducer", AlwaysFailsProducer)
        monkeypatch.setattr(exporters.time, "sleep", lambda *_: None)
        with pytest.raises(RuntimeError, match="Kafka"):
            exporters.export_kafka(small_graph, bootstrap="fake:9092")


class TestExportNeo4j:
    def test_wipes_and_loads_accounts_and_transactions(self, small_graph, monkeypatch):
        neo4j = pytest.importorskip("neo4j")
        calls = []

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def run(self, query, **kw):
                calls.append((" ".join(query.split()), kw.get("rows", None)))  # collapse whitespace, keep full text

        class FakeDriver:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def verify_connectivity(self):
                pass

            def session(self):
                return FakeSession()

        monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **kw: FakeDriver())
        exporters.export_neo4j(small_graph, uri="bolt://fake:7687", user="u", password="p", batch=100, wipe=True)
        assert any("DETACH DELETE" in q for q, _ in calls)
        assert any("CONSTRAINT" in q for q, _ in calls)
        assert any("MERGE (a:Account" in q for q, _ in calls)
        assert any("MATCH (s:Account" in q for q, _ in calls)

    def test_skips_wipe_when_wipe_is_false(self, small_graph, monkeypatch):
        neo4j = pytest.importorskip("neo4j")
        calls = []

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def run(self, query, **kw):
                calls.append(" ".join(query.split()))

        class FakeDriver:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def verify_connectivity(self):
                pass

            def session(self):
                return FakeSession()

        monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **kw: FakeDriver())
        exporters.export_neo4j(small_graph, uri="bolt://fake:7687", user="u", password="p", wipe=False)
        assert not any("DETACH DELETE" in q for q in calls)

    def test_raises_if_never_reachable(self, small_graph, monkeypatch):
        neo4j = pytest.importorskip("neo4j")

        class UnreachableDriver:
            def verify_connectivity(self):
                raise OSError("simulated: connection refused")

        monkeypatch.setattr(neo4j.GraphDatabase, "driver", lambda *a, **kw: UnreachableDriver())
        monkeypatch.setattr(exporters.time, "sleep", lambda *_: None)
        with pytest.raises(RuntimeError, match="Neo4j"):
            exporters.export_neo4j(small_graph, uri="bolt://fake:7687", user="u", password="p")
