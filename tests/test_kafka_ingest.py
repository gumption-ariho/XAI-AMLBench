"""Tests for gnn_aml_core.kafka_ingest -- the Option 2 live-feed consumer.

No real Kafka broker or network needed for any of this: the `kafka` package is only ever imported lazily,
inside _make_consumer(), so the module itself (and everything below except _make_consumer and run()'s use of
a real consumer) imports and runs with nothing more than the standard library. _ingest_batch's HTTP call to
backend is exercised by monkeypatching urllib.request.urlopen rather than hitting a real server. run()'s
batching/flush logic is exercised with a fake consumer object (a plain iterable of fake messages) in place of
a real KafkaConsumer, since that's the only seam run() actually needs.
"""
import json

import pytest

import gnn_aml_core.kafka_ingest as ki


def test_module_imports_without_the_kafka_package_installed():
    # kafka_ingest.py must stay importable (e.g. for this test module itself, or anything that introspects it)
    # even on a machine that hasn't installed the `kafka-python` package yet -- only actually running the
    # consumer loop needs it, and that import is deferred into _make_consumer() for exactly that reason.
    assert hasattr(ki, "_make_consumer")
    assert hasattr(ki, "run")


class TestRowFromMessage:
    def test_minimal_valid_message(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 12.5})
        assert row == {"src": "A", "dst": "B", "amount": 12.5}

    def test_missing_src_is_skipped(self):
        assert ki._row_from_message({"dst": "B", "amount": 1}) is None

    def test_missing_dst_is_skipped(self):
        assert ki._row_from_message({"src": "A", "amount": 1}) is None

    def test_blank_src_is_skipped(self):
        assert ki._row_from_message({"src": "   ", "dst": "B", "amount": 1}) is None

    def test_missing_amount_is_skipped(self):
        assert ki._row_from_message({"src": "A", "dst": "B"}) is None

    def test_non_numeric_amount_is_skipped(self):
        assert ki._row_from_message({"src": "A", "dst": "B", "amount": "not-a-number"}) is None

    def test_amount_given_as_numeric_string_is_coerced(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": "99.9"})
        assert row["amount"] == 99.9

    def test_src_and_dst_are_stripped_and_stringified(self):
        row = ki._row_from_message({"src": " A ", "dst": 42, "amount": 1})
        assert row["src"] == "A" and row["dst"] == "42"

    def test_bank_field_is_carried_through(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "bank": "first-national"})
        assert row["bank"] == "first-national"

    def test_institution_field_is_used_when_bank_is_absent(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "institution": "second-bank"})
        assert row["bank"] == "second-bank"

    def test_bank_takes_priority_over_institution_when_both_present(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "bank": "x", "institution": "y"})
        assert row["bank"] == "x"

    def test_default_bank_env_var_is_used_when_message_carries_neither(self, monkeypatch):
        monkeypatch.setattr(ki, "LIVE_FEED_DEFAULT_BANK", "fallback-bank")
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1})
        assert row["bank"] == "fallback-bank"

    def test_no_bank_key_at_all_when_nothing_identifies_one(self, monkeypatch):
        monkeypatch.setattr(ki, "LIVE_FEED_DEFAULT_BANK", "")
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1})
        assert "bank" not in row

    def test_timestamp_is_coerced_to_int_when_present(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "timestamp": "1700000000"})
        assert row["timestamp"] == 1700000000

    def test_timestamp_is_omitted_when_blank(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "timestamp": ""})
        assert "timestamp" not in row

    def test_timestamp_is_omitted_when_unparseable(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "timestamp": "not-a-time"})
        assert "timestamp" not in row

    def test_cross_border_is_coerced_to_int_when_present(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "cross_border": "1"})
        assert row["cross_border"] == 1

    def test_cross_border_is_omitted_when_blank(self):
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "cross_border": None})
        assert "cross_border" not in row

    def test_ignores_fields_the_detector_must_never_read(self):
        # is_laundering/typology/pattern_id would answer the exact question this system exists to answer on
        # its own -- a real bank feed wouldn't carry them anyway, but a synthetic feed (aml_synth's own
        # exporter) does, so this is the one place that MUST NOT read them even though they're right there.
        row = ki._row_from_message({"src": "A", "dst": "B", "amount": 1, "is_laundering": True,
                                     "typology": "layering", "pattern_id": "p-1"})
        assert set(row.keys()) == {"src", "dst", "amount"}


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestIngestBatch:
    def test_posts_rows_as_json_body_to_live_ingest_endpoint(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            captured["headers"] = dict(req.header_items())
            captured["timeout"] = timeout
            return _FakeResponse({"n_accounts": 3, "n_transactions": 5, "alerts_created": []})

        monkeypatch.setattr(ki.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(ki, "BACKEND_API_URL", "http://backend:8000/api")
        monkeypatch.setattr(ki, "GNN_ADMIN_TOKEN", "")

        rows = [{"src": "A", "dst": "B", "amount": 1.0}]
        result = ki._ingest_batch(rows)

        assert captured["url"] == "http://backend:8000/api/system/live-ingest"
        assert captured["body"] == {"rows": rows}
        assert result["n_accounts"] == 3

    def test_trailing_slash_on_backend_url_does_not_double_up(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _FakeResponse({"n_accounts": 0, "n_transactions": 0, "alerts_created": []})

        monkeypatch.setattr(ki.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(ki, "BACKEND_API_URL", "http://backend:8000/api/")
        monkeypatch.setattr(ki, "GNN_ADMIN_TOKEN", "")

        ki._ingest_batch([])
        assert captured["url"] == "http://backend:8000/api/system/live-ingest"

    def test_admin_token_is_sent_as_bearer_header_when_set(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.header_items())
            return _FakeResponse({"n_accounts": 0, "n_transactions": 0, "alerts_created": []})

        monkeypatch.setattr(ki.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(ki, "BACKEND_API_URL", "http://backend:8000/api")
        monkeypatch.setattr(ki, "GNN_ADMIN_TOKEN", "secret-token")

        ki._ingest_batch([])
        assert captured["headers"].get("Authorization") == "Bearer secret-token"

    def test_no_authorization_header_when_admin_token_is_blank(self, monkeypatch):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["headers"] = dict(req.header_items())
            return _FakeResponse({"n_accounts": 0, "n_transactions": 0, "alerts_created": []})

        monkeypatch.setattr(ki.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(ki, "BACKEND_API_URL", "http://backend:8000/api")
        monkeypatch.setattr(ki, "GNN_ADMIN_TOKEN", "")

        ki._ingest_batch([])
        assert "Authorization" not in captured["headers"]


class _FakeMessage:
    def __init__(self, value):
        self.value = value


class _FakeConsumer:
    """Stands in for kafka.KafkaConsumer: iterating it yields a fixed batch of fake messages once, then raises
    StopIteration (like the real consumer_timeout_ms=1000 behaviour when the topic goes idle). Each call to
    run()'s `for message in consumer:` loop gets the next chunk in `chunks`, same shape as repeated polls."""

    def __init__(self, chunks: list[list]):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        if self._chunks:
            yield from self._chunks.pop(0)
        return

    def close(self):
        self.closed = True


class TestRun:
    """run()'s own logic -- batching by count, flushing, skip-counting, and the consumer being closed on exit
    -- using a fake consumer and a fake clock so none of this depends on real time or a real broker."""

    def _run_until_raises(self, monkeypatch, chunks, batch_size=25, max_wait_seconds=5.0,
                          ingest_results=None, clock_values=None):
        fake_consumer = _FakeConsumer(chunks)
        monkeypatch.setattr(ki, "_make_consumer", lambda: fake_consumer)

        ingest_calls = []
        ingest_results = list(ingest_results or [])

        def fake_ingest_batch(rows):
            ingest_calls.append(list(rows))
            if ingest_results:
                res = ingest_results.pop(0)
                if isinstance(res, Exception):
                    raise res
                return res
            return {"n_accounts": 0, "n_transactions": 0, "accounts_scanned": 0, "alerts_created": []}

        monkeypatch.setattr(ki, "_ingest_batch", fake_ingest_batch)

        clock = {"t": 0.0}
        clock_values = list(clock_values or [])

        def fake_monotonic():
            if clock_values:
                clock["t"] = clock_values.pop(0)
            return clock["t"]

        monkeypatch.setattr(ki.time, "monotonic", fake_monotonic)

        # run() loops forever; once the fake consumer has no more chunks left to yield and the deadline isn't
        # due, the outer `while True:` would spin forever in a real test. We stop it deterministically by
        # having the *second* empty iteration raise, instead of sleeping or using a thread/timeout.
        call_count = {"n": 0}
        real_iter = _FakeConsumer.__iter__

        def counting_iter(self):
            call_count["n"] += 1
            if call_count["n"] > len(chunks) + 1:
                raise _StopRun()
            yield from real_iter(self)

        monkeypatch.setattr(_FakeConsumer, "__iter__", counting_iter)

        with pytest.raises(_StopRun):
            ki.run(batch_size=batch_size, max_wait_seconds=max_wait_seconds)

        return fake_consumer, ingest_calls

    def test_batch_flushes_as_soon_as_batch_size_is_reached(self, monkeypatch):
        chunk = [_FakeMessage({"src": "A", "dst": "B", "amount": 1}),
                 _FakeMessage({"src": "C", "dst": "D", "amount": 2})]
        _, calls = self._run_until_raises(monkeypatch, [chunk], batch_size=2, clock_values=[0.0, 0.0, 0.0])
        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_malformed_messages_are_skipped_not_fatal(self, monkeypatch):
        chunk = [_FakeMessage({"src": "A", "dst": "B", "amount": 1}),
                 _FakeMessage({"src": "", "dst": "B", "amount": 1}),  # malformed: blank src
                 _FakeMessage({"src": "C", "dst": "D", "amount": 2})]
        _, calls = self._run_until_raises(monkeypatch, [chunk], batch_size=2, clock_values=[0.0, 0.0, 0.0])
        # only the 2 well-formed rows ever reach a flush; the malformed one is silently dropped, not counted
        # as a transaction, and never crashes the loop
        assert len(calls) == 1
        assert len(calls[0]) == 2

    def test_time_based_flush_fires_even_below_batch_size(self, monkeypatch):
        chunk = [_FakeMessage({"src": "A", "dst": "B", "amount": 1})]  # only 1 message, batch_size is 25
        # deadline check happens after the for-loop drains the chunk; make monotonic() report we're past it
        _, calls = self._run_until_raises(monkeypatch, [chunk], batch_size=25,
                                          clock_values=[0.0, 999.0, 999.0, 999.0])
        assert len(calls) == 1
        assert len(calls[0]) == 1

    def test_a_failed_flush_does_not_crash_the_loop(self, monkeypatch):
        import urllib.error
        chunk = [_FakeMessage({"src": "A", "dst": "B", "amount": 1})]
        # _ingest_batch raising URLError must be swallowed (logged, batch dropped) -- the consumer loop keeps
        # running and the next iteration starts a fresh, empty batch rather than taking the process down
        _, calls = self._run_until_raises(monkeypatch, [chunk], batch_size=25,
                                          ingest_results=[urllib.error.URLError("backend unreachable")],
                                          clock_values=[0.0, 999.0, 999.0, 999.0])
        assert len(calls) == 1  # the attempt was made and failed -- the loop itself didn't die

    def test_consumer_is_closed_when_the_loop_exits(self, monkeypatch):
        # run()'s own `finally: consumer.close()` must fire even on an unexpected exception bubbling out of
        # the loop (here, our deterministic _StopRun stand-in for "the process is being shut down")
        chunk = [_FakeMessage({"src": "A", "dst": "B", "amount": 1})]
        consumer, _ = self._run_until_raises(monkeypatch, [chunk], batch_size=25, clock_values=[0.0] * 10)
        assert consumer.closed is True


class _StopRun(Exception):
    """A deterministic stand-in for 'the process is being shut down' -- used only to end run()'s infinite
    loop at an exact, predictable point in these tests, instead of sleeping or using a background thread."""


class TestMain:
    def test_parses_args_and_calls_run_with_them(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(ki, "run", lambda **kw: captured.update(kw))
        monkeypatch.setattr("sys.argv", ["kafka_ingest", "--batch-size", "10", "--max-wait-seconds", "2.5"])
        ki.main()
        assert captured == {"batch_size": 10, "max_wait_seconds": 2.5}

    def test_defaults_come_from_environment_variables(self, monkeypatch):
        captured = {}
        monkeypatch.setenv("LIVE_FEED_BATCH_SIZE", "40")
        monkeypatch.setenv("LIVE_FEED_MAX_WAIT_SECONDS", "9.0")
        monkeypatch.setattr(ki, "run", lambda **kw: captured.update(kw))
        monkeypatch.setattr("sys.argv", ["kafka_ingest"])
        ki.main()
        assert captured == {"batch_size": 40, "max_wait_seconds": 9.0}
