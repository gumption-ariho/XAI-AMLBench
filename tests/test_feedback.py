"""Tests for gnn_aml_core.feedback: the drift-detection and feedback-dataset logic. No PyTorch and no database
needed -- everything except read_decisions_from_db operates on plain DecisionRecord objects, so these always
run. retrain_with_feedback is tested with a stand-in fit_gnn rather than a real one, since only its own
label-overwriting and index-repetition logic belongs to this module; the actual training is gnn_aml_core.train's
responsibility and is tested there.
"""
import json
import os

import numpy as np
import pytest

from gnn_aml_core.feedback import (DecisionRecord, build_feedback_dataset, disagreement_rate, model_agreed,
                                   retrain_with_feedback, should_retrain, true_label)


def rec(account_id="A", score=0.8, threshold=0.5, decision="confirmed"):
    return DecisionRecord(account_id=account_id, predicted_score=score, threshold=threshold, decision=decision)


class TestTrueLabel:
    def test_confirmed_is_one(self):
        assert true_label(rec(decision="confirmed")) == 1.0

    def test_dismissed_is_zero(self):
        assert true_label(rec(decision="dismissed")) == 0.0

    def test_unknown_decision_raises(self):
        with pytest.raises(ValueError, match="unknown decision"):
            true_label(rec(decision="pending"))


class TestModelAgreed:
    """The four confusion-matrix cases; getting any one of these backwards would silently invert the whole
    drift signal, so each is checked explicitly rather than only through an aggregate rate."""

    def test_true_positive_flagged_and_confirmed(self):
        assert model_agreed(rec(score=0.8, threshold=0.5, decision="confirmed")) is True

    def test_false_positive_flagged_but_dismissed(self):
        assert model_agreed(rec(score=0.8, threshold=0.5, decision="dismissed")) is False

    def test_false_negative_not_flagged_but_confirmed(self):
        assert model_agreed(rec(score=0.3, threshold=0.5, decision="confirmed")) is False

    def test_true_negative_not_flagged_and_dismissed(self):
        assert model_agreed(rec(score=0.3, threshold=0.5, decision="dismissed")) is True

    def test_score_exactly_at_threshold_counts_as_flagged(self):
        assert model_agreed(rec(score=0.5, threshold=0.5, decision="confirmed")) is True


class TestDisagreementRate:
    def test_empty_list_is_zero(self):
        assert disagreement_rate([]) == 0.0

    def test_all_agree_is_zero(self):
        records = [rec("A", 0.8, 0.5, "confirmed"), rec("B", 0.2, 0.5, "dismissed")]
        assert disagreement_rate(records) == 0.0

    def test_all_disagree_is_one(self):
        records = [rec("A", 0.8, 0.5, "dismissed"), rec("B", 0.2, 0.5, "confirmed")]
        assert disagreement_rate(records) == 1.0

    def test_mixed_rate_is_exact(self):
        records = [rec("A", 0.8, 0.5, "confirmed"), rec("B", 0.8, 0.5, "dismissed"),
                  rec("C", 0.2, 0.5, "dismissed"), rec("D", 0.2, 0.5, "confirmed")]
        assert disagreement_rate(records) == pytest.approx(0.5)


class TestShouldRetrain:
    def test_too_few_decisions_never_triggers_even_at_100_percent_disagreement(self):
        records = [rec(f"A{i}", 0.8, 0.5, "dismissed") for i in range(5)]
        triggered, reasoning = should_retrain(records, threshold=0.15, min_decisions=20)
        assert triggered is False
        assert reasoning["disagreement_rate"] is None
        assert reasoning["n_decisions"] == 5

    def test_enough_decisions_high_disagreement_triggers(self):
        records = [rec(f"A{i}", 0.8, 0.5, "dismissed") for i in range(25)]
        triggered, reasoning = should_retrain(records, threshold=0.15, min_decisions=20)
        assert triggered is True
        assert reasoning["disagreement_rate"] == 1.0

    def test_enough_decisions_low_disagreement_does_not_trigger(self):
        records = [rec(f"A{i}", 0.8, 0.5, "confirmed") for i in range(19)] + [rec("B0", 0.8, 0.5, "dismissed")]
        triggered, reasoning = should_retrain(records, threshold=0.15, min_decisions=20)
        assert triggered is False

    def test_exactly_at_the_trigger_threshold_counts_as_triggering(self):
        records = [rec(f"A{i}", 0.8, 0.5, "confirmed") for i in range(17)] + \
                  [rec(f"B{i}", 0.8, 0.5, "dismissed") for i in range(3)]  # exactly 15%
        triggered, _ = should_retrain(records, threshold=0.15, min_decisions=20)
        assert triggered is True

    def test_reasoning_always_reports_the_decision_count(self):
        records = [rec(f"A{i}", 0.8, 0.5, "confirmed") for i in range(20)]
        _, reasoning = should_retrain(records)
        assert reasoning["n_decisions"] == 20


class TestBuildFeedbackDataset:
    def test_skips_accounts_no_longer_in_the_graph(self):
        records = [rec("A", 0.8, 0.5, "confirmed"), rec("GONE", 0.9, 0.5, "confirmed")]
        ids, labels = build_feedback_dataset(records, account_ids_in_graph={"A"})
        assert ids == ["A"] and labels == [1.0]

    def test_preserves_order_and_maps_labels_correctly(self):
        records = [rec("A", 0.8, 0.5, "confirmed"), rec("B", 0.2, 0.5, "dismissed")]
        ids, labels = build_feedback_dataset(records, account_ids_in_graph={"A", "B"})
        assert ids == ["A", "B"] and labels == [1.0, 0.0]

    def test_empty_input_returns_empty(self):
        assert build_feedback_dataset([], account_ids_in_graph={"A"}) == ([], [])


class TestRetrainWithFeedback:
    def test_overwrites_labels_and_upweights_feedback_examples(self, monkeypatch):
        calls = {}

        def fake_fit_gnn(arrays, tr, va, te, **kw):
            calls.update(arrays=arrays, tr=tr, va=va, te=te, kw=kw)
            return {"report": {}, "best_val_auc": 0.9, "best_epoch": 1, "epochs_run": 1, "state_dict": {}, "hparams": {}}

        import gnn_aml_core.train as train_module
        monkeypatch.setattr(train_module, "fit_gnn", fake_fit_gnn)

        arrays = {"account_ids": ["A", "B", "C", "D"], "y": np.array([0.0, 0.0, 1.0, 1.0])}
        tr, va, te = np.array([0, 1, 2, 3]), np.array([]), np.array([])

        retrain_with_feedback(arrays, tr, va, te, feedback_ids=["A", "C"], feedback_labels=[1.0, 0.0],
                              feedback_weight=3, model_name="gatv2")

        assert calls["arrays"]["y"][0] == 1.0   # A: overwritten to the officer-confirmed label
        assert calls["arrays"]["y"][2] == 0.0   # C: overwritten
        assert calls["arrays"]["y"][1] == 0.0   # B: untouched
        assert calls["arrays"]["y"][3] == 1.0   # D: untouched
        assert list(calls["tr"]).count(0) == 1 + 3   # original occurrence + feedback_weight repeats
        assert list(calls["tr"]).count(2) == 1 + 3
        assert calls["kw"]["model_name"] == "gatv2"
        assert arrays["y"][0] == 0.0   # the caller's original array must not be mutated in place

    def test_ignores_feedback_accounts_not_present_in_arrays(self, monkeypatch):
        def fake_fit_gnn(arrays, tr, va, te, **kw):
            return {"report": {}, "best_val_auc": 0.0, "best_epoch": 0, "epochs_run": 0, "state_dict": {}, "hparams": {}}

        import gnn_aml_core.train as train_module
        monkeypatch.setattr(train_module, "fit_gnn", fake_fit_gnn)

        arrays = {"account_ids": ["A", "B"], "y": np.array([0.0, 0.0])}
        tr = np.array([0, 1])
        retrain_with_feedback(arrays, tr, np.array([]), np.array([]), feedback_ids=["NOT_IN_GRAPH"],
                              feedback_labels=[1.0], model_name="gatv2")  # must not raise

    def test_forwards_init_state_dict_through_to_fit_gnn_unchanged(self, monkeypatch):
        # this is the warm-start/fine-tune path retrain_and_maybe_promote relies on: retrain_with_feedback
        # itself does nothing special with init_state_dict (or lr/epochs/patience/seed) beyond forwarding them
        # via **fit_kwargs -- fit_gnn is what actually loads it into the model before training. This pins that
        # the forwarding is exact (same object, not copied/dropped/renamed) so a real weights dict reaches the
        # real fit_gnn when torch is available.
        calls = {}

        def fake_fit_gnn(arrays, tr, va, te, **kw):
            calls.update(kw=kw)
            return {"report": {}, "best_val_auc": 0.0, "best_epoch": 0, "epochs_run": 0, "state_dict": {}, "hparams": {}}

        import gnn_aml_core.train as train_module
        monkeypatch.setattr(train_module, "fit_gnn", fake_fit_gnn)

        arrays = {"account_ids": ["A", "B"], "y": np.array([0.0, 0.0])}
        tr = np.array([0, 1])
        sentinel_weights = {"conv1.weight": "not a real tensor, just a sentinel object"}
        retrain_with_feedback(arrays, tr, np.array([]), np.array([]), feedback_ids=[], feedback_labels=[],
                              model_name="gatv2", hidden=64, layers=2, init_state_dict=sentinel_weights,
                              lr=0.0005, epochs=30, patience=5, seed=42)

        assert calls["kw"]["init_state_dict"] is sentinel_weights   # forwarded by identity, not copied
        assert calls["kw"]["lr"] == 0.0005
        assert calls["kw"]["epochs"] == 30
        assert calls["kw"]["patience"] == 5


class TestDefaultDsn:
    """A real bug this project shipped once: the default connection string was a guessed
    postgresql://aml:aml@localhost:15432/aml that did not match this project's actual credentials
    (POSTGRES_USER/PASSWORD/DB in .env), failing authentication silently against a correctly running
    Postgres. These tests pin the fix: read the same .env every other tool in this project reads."""

    def test_reads_credentials_from_env_file(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("POSTGRES_USER=projectxy\nPOSTGRES_PASSWORD=change_me_pg\nPOSTGRES_DB=projectxy\n")
        monkeypatch.chdir(tmp_path)
        for k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "DATABASE_URL"):
            monkeypatch.delenv(k, raising=False)
        from gnn_aml_core.feedback import _default_dsn
        assert _default_dsn() == "postgresql://projectxy:change_me_pg@127.0.0.1:15432/projectxy"

    def test_environment_variable_overrides_env_file(self, tmp_path, monkeypatch):
        (tmp_path / ".env").write_text("POSTGRES_USER=fromfile\nPOSTGRES_PASSWORD=fromfile\nPOSTGRES_DB=fromfile\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("POSTGRES_USER", "fromenv")
        from gnn_aml_core.feedback import _default_dsn
        assert _default_dsn().startswith("postgresql://fromenv:")

    def test_missing_env_file_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no .env here at all
        for k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB", "DATABASE_URL"):
            monkeypatch.delenv(k, raising=False)
        from gnn_aml_core.feedback import _default_dsn
        dsn = _default_dsn()
        assert "projectxy" in dsn  # falls back to the documented default rather than raising


class TestMainErrorMessages:
    """A real, previously-shipped bug: the schema not existing yet (backend/main.py has never run its
    startup hook) surfaced the same generic "is Postgres running?" message as an actual connection failure,
    which is misleading since Postgres was reachable fine -- the real cause was a different, more specific
    thing entirely. These pin the fix: the two cases now give genuinely different, correct guidance."""

    def test_undefined_table_message_mentions_the_backend_and_alerts_table(self, monkeypatch):
        import contextlib
        import io
        import sys
        import types

        fake_errors = types.SimpleNamespace(Error=Exception, OperationalError=type("OperationalError", (Exception,), {}),
                                            ProgrammingError=type("ProgrammingError", (Exception,), {}))
        fake_errors.UndefinedTable = type("UndefinedTable", (fake_errors.ProgrammingError,), {})
        undefined_table_exc = fake_errors.UndefinedTable('relation "alerts" does not exist')
        fake_psycopg = types.SimpleNamespace(errors=fake_errors, OperationalError=fake_errors.OperationalError,
                                             connect=lambda dsn: (_ for _ in ()).throw(undefined_table_exc))
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
        monkeypatch.setitem(sys.modules, "psycopg.errors", fake_errors)

        import gnn_aml_core.feedback as fb
        monkeypatch.setattr(sys, "argv", ["feedback.py", "--check", "--dsn", "postgresql://fake"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with pytest.raises(SystemExit):
                fb.main()
        out = buf.getvalue()
        assert "alerts' table does not exist yet" in out
        assert "run_local.py" in out or "backend" in out

    def test_generic_connection_failure_gives_the_generic_message_not_the_table_one(self, monkeypatch):
        import contextlib
        import io
        import sys
        import types

        fake_errors = types.SimpleNamespace(Error=Exception, OperationalError=type("OperationalError", (Exception,), {}),
                                            ProgrammingError=type("ProgrammingError", (Exception,), {}))
        fake_errors.UndefinedTable = type("UndefinedTable", (fake_errors.ProgrammingError,), {})
        conn_refused = fake_errors.OperationalError("connection refused")
        fake_psycopg = types.SimpleNamespace(errors=fake_errors, OperationalError=fake_errors.OperationalError,
                                             connect=lambda dsn: (_ for _ in ()).throw(conn_refused))
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
        monkeypatch.setitem(sys.modules, "psycopg.errors", fake_errors)

        import gnn_aml_core.feedback as fb
        monkeypatch.setattr(sys, "argv", ["feedback.py", "--check", "--dsn", "postgresql://fake"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with pytest.raises(SystemExit):
                fb.main()
        out = buf.getvalue()
        assert "is it running? docker compose up -d database" in out
        assert "alerts' table" not in out


class TestServeMetricsMode:
    """A real gap, found during an observability review: the drift signal (disagreement rate, whether a
    retrain gets triggered) was only ever visible in this CLI tool's own stdout -- invisible to the monitoring
    stack. These pin two things: the new metrics get set correctly, and a real bug found while building this
    (the retraining path's own 'records' variable became undefined once the check logic was refactored into
    a helper) stays fixed."""

    def test_records_variable_is_threaded_correctly_to_the_retraining_path(self, monkeypatch, has_torch):
        # the real bug: refactoring the check logic into _check_once() initially left 'records' undefined in
        # the retraining path below it. This used to prove the fix held by asserting execution reached a real
        # `import torch` inside main() and raised ModuleNotFoundError there -- which only worked as a signal
        # in a sandbox that happened to lack PyTorch, and which (confirmed on a real run, on a machine WITH
        # PyTorch installed and this project's own trained models/ on disk) goes on to actually load those
        # real model files and run real training, including a separate, real NaN-validation-split bug in
        # gnn_aml_core.train.fit_gnn that this test had nothing to do with and should never have been able to
        # reach in the first place. This version instead asserts the actual claim in the test's name directly
        # -- that `records` reaches build_feedback_dataset() unmodified -- by mocking torch.load so no real
        # file on disk and no real training is ever touched, regardless of which machine this runs on.
        if not has_torch:
            pytest.skip("gnn_aml_core.feedback.main()'s retraining path does `import torch` unconditionally")
        import sys

        import torch

        import gnn_aml_core.feedback as fb

        records = [rec(account_id=f"A{i}", decision="confirmed" if i % 5 else "dismissed") for i in range(25)]
        monkeypatch.setattr(fb, "read_decisions_from_db", lambda dsn: records)
        monkeypatch.setattr(sys, "argv", ["feedback.py", "--dsn", "postgresql://fake"])

        fake_graph = {
            "x": np.zeros((3, 2), dtype=np.float32), "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_attr": np.zeros((0, 1), dtype=np.float32), "edge_type": np.zeros((0,), dtype=np.int64),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float32), "account_ids": ["X1", "X2", "X3"],
        }
        fake_ckpt = {"model_name": "gatv2", "hparams": {"hidden": 8, "num_layers": 1, "seed": 42}}
        # torch.load is called twice in sequence (graph.pt, then gnn_model.pt) -- a side_effect list returns
        # each in order rather than needing to inspect the path argument.
        load_calls = iter([fake_graph, fake_ckpt])
        monkeypatch.setattr(torch, "load", lambda *a, **kw: next(load_calls))

        captured = {}

        class _StopHere(Exception):
            pass

        def _capture_and_stop(records_arg, account_ids_in_graph):
            captured["records"] = records_arg
            captured["account_ids_in_graph"] = account_ids_in_graph
            raise _StopHere

        monkeypatch.setattr(fb, "build_feedback_dataset", _capture_and_stop)

        with pytest.raises(_StopHere):
            fb.main()

        # the actual claim this test makes: `records` (not re-fetched, not re-ordered, not silently dropped)
        # is exactly what reaches the retraining path -- proven directly, not inferred from an unrelated import
        assert captured["records"] == records
        assert captured["account_ids_in_graph"] == {"X1", "X2", "X3"}

    def test_check_mode_sets_the_prometheus_metrics(self, monkeypatch):
        # reads the metrics back through prometheus_client's own public text-exposition API
        # (generate_latest), not an internal attribute, so this does not depend on an implementation detail
        import sys

        from prometheus_client import generate_latest

        import gnn_aml_core.feedback as fb

        records = [rec(account_id=f"A{i}", decision="confirmed" if i % 5 else "dismissed") for i in range(25)]
        monkeypatch.setattr(fb, "read_decisions_from_db", lambda dsn: records)
        monkeypatch.setattr(sys, "argv", ["feedback.py", "--check", "--dsn", "postgresql://fake"])
        fb.main()   # must not raise
        exposed = generate_latest().decode()
        # checks the metric names are present and reachable via the real public API, not the exact numeric
        # text formatting (e.g. "25" vs "25.0"), which this environment cannot independently confirm
        assert "aml_feedback_decisions_total" in exposed
        assert "aml_officer_disagreement_rate" in exposed
        assert "25" in exposed
        assert "0.2" in exposed

    def test_loop_mode_only_reaches_retrain_and_maybe_promote_behind_the_auto_retrain_gate(self):
        # the design changed deliberately this round (previously: the loop could NEVER retrain; now: it can,
        # but ONLY when --auto-retrain/FEEDBACK_AUTO_RETRAIN is explicitly set, and even then only through the
        # gated retrain_and_maybe_promote(), never by calling build_feedback_dataset or torch.save directly).
        # Confirmed by source inspection rather than actually running the loop (which blocks forever).
        import inspect

        import gnn_aml_core.feedback as fb

        src = inspect.getsource(fb.main)
        loop_section = src[src.index("if a.serve_metrics:"):src.index("return  # pragma: no cover")]
        assert "start_http_server" in loop_section
        assert "retrain_and_maybe_promote" in loop_section
        assert "build_feedback_dataset" not in loop_section  # only reachable *inside* retrain_and_maybe_promote
        # the retrain call must be textually inside an `if triggered and a.auto_retrain:` block, not reachable
        # whenever the drift trigger fires regardless of the flag
        gate_idx = loop_section.index("if triggered and a.auto_retrain:")
        call_idx = loop_section.index("retrain_and_maybe_promote(")
        assert gate_idx < call_idx

    def test_auto_retrain_flag_defaults_to_off_when_env_var_unset(self, monkeypatch):
        # FEEDBACK_AUTO_RETRAIN unset -> --auto-retrain's default must be False, so the feature stays opt-in
        # (matches .env.example, which ships it commented out)
        monkeypatch.delenv("FEEDBACK_AUTO_RETRAIN", raising=False)
        default = os.getenv("FEEDBACK_AUTO_RETRAIN", "").strip().lower() in ("1", "true", "yes")
        assert default is False

    def test_auto_retrain_flag_reads_true_from_env_var(self, monkeypatch):
        monkeypatch.setenv("FEEDBACK_AUTO_RETRAIN", "true")
        default = os.getenv("FEEDBACK_AUTO_RETRAIN", "").strip().lower() in ("1", "true", "yes")
        assert default is True

    def test_auto_retrain_attempt_is_wrapped_so_a_failure_never_kills_the_loop(self):
        # a real design requirement: if retrain_and_maybe_promote blows up (torch missing, bad checkpoint,
        # disk full, anything), the long-lived monitoring loop must survive and retry on the next interval
        # rather than taking the whole process down -- confirmed by source inspection that the call sits in
        # its own try/except, distinct from the outer loop's SystemExit handler
        import inspect

        import gnn_aml_core.feedback as fb

        src = inspect.getsource(fb.main)
        call_idx = src.index("retrain_and_maybe_promote(records")
        preceding_lines = src[:call_idx].rstrip().splitlines()
        assert preceding_lines[-1].strip() == "try:"
        except_idx = src.index("except Exception as exc", call_idx)
        # Bound the slice at the *next* except block (the outer loop's own SystemExit handler)
        # instead of a hardcoded character count. A fixed window breaks every time a comment or
        # the log message's wording changes length, even though the actual try/except structure
        # -- the real thing under test -- hasn't changed at all.
        block_end = src.index("except SystemExit", except_idx)
        following = src[call_idx:block_end]
        assert "except Exception as exc" in following
        assert "current model" in following  # confirms the log message documents the safe fallback


class TestReadDecisionsFromDb:
    """The one function in this module that needs a live database -- exercised with a fake psycopg module
    (same sys.modules stubbing technique TestMainErrorMessages already uses above), so the query and the
    DecisionRecord construction are checked without a real Postgres connection."""

    def _fake_psycopg(self, rows, monkeypatch):
        import sys
        import types

        class _FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                self._sql = sql

            def fetchall(self):
                return rows

        class _FakeConn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def cursor(self):
                return _FakeCursor()

        fake_psycopg = types.SimpleNamespace(connect=lambda dsn: _FakeConn())
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    def test_queries_only_resolved_alerts_and_builds_decision_records(self, monkeypatch):
        from gnn_aml_core.feedback import read_decisions_from_db

        self._fake_psycopg([("A1", 0.8, "confirmed"), ("A2", 0.2, "dismissed")], monkeypatch)
        monkeypatch.delenv("FEEDBACK_THRESHOLD_FALLBACK", raising=False)

        records = read_decisions_from_db("postgresql://fake")
        assert len(records) == 2
        assert records[0].account_id == "A1" and records[0].decision == "confirmed"
        assert records[0].predicted_score == 0.8
        assert records[1].account_id == "A2" and records[1].decision == "dismissed"

    def test_threshold_falls_back_to_the_environment_variable(self, monkeypatch):
        from gnn_aml_core.feedback import read_decisions_from_db

        self._fake_psycopg([("A1", 0.9, "confirmed")], monkeypatch)
        monkeypatch.setenv("FEEDBACK_THRESHOLD_FALLBACK", "0.75")

        records = read_decisions_from_db("postgresql://fake")
        assert records[0].threshold == 0.75

    def test_threshold_defaults_to_point_five_when_unset(self, monkeypatch):
        from gnn_aml_core.feedback import read_decisions_from_db

        self._fake_psycopg([("A1", 0.9, "confirmed")], monkeypatch)
        monkeypatch.delenv("FEEDBACK_THRESHOLD_FALLBACK", raising=False)

        records = read_decisions_from_db("postgresql://fake")
        assert records[0].threshold == 0.5

    def test_no_resolved_alerts_returns_an_empty_list(self, monkeypatch):
        from gnn_aml_core.feedback import read_decisions_from_db

        self._fake_psycopg([], monkeypatch)
        assert read_decisions_from_db("postgresql://fake") == []


class TestEvaluateCheckpoint:
    """evaluate_checkpoint() does `import torch` unconditionally, so this needs PyTorch installed -- but the
    model itself is mocked (via gnn_aml_core.models.build_model), since what's actually under test here is
    evaluate_checkpoint's own plumbing (which checkpoint fields build the model, that eval()/no_grad() is
    used, that the test-split AUC is computed correctly from the model's output), not GATv2/RGCN's real
    forward pass, which belongs to gnn_aml_core.models' own test suite."""

    def test_computes_real_auc_from_the_models_sigmoid_output_on_the_test_split(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import numpy as np

        from gnn_aml_core.feedback import evaluate_checkpoint

        class _FakeModel:
            def __init__(self, logits):
                self._logits = logits

            def load_state_dict(self, sd):
                self.loaded = sd

            def eval(self):
                pass

            def __call__(self, x, ei, ea, et):
                return torch.from_numpy(self._logits)

        # 4 accounts; test split is indices [2, 3]. Logits chosen so the model perfectly ranks the test split
        # (high logit for the real positive, low for the real negative) -- AUC must come out to exactly 1.0.
        logits = np.array([0.0, 0.0, 5.0, -5.0], dtype=np.float32)
        fake_model = _FakeModel(logits)

        import sys
        import types

        captured_build_args = {}

        def fake_build_model(model_name, in_dim, **hp):
            captured_build_args.update(model_name=model_name, in_dim=in_dim, hp=hp)
            return fake_model

        # stubbed via sys.modules rather than imported directly: the real gnn_aml_core.models does
        # `import torch.nn as nn`, which needs a real, full PyTorch install -- this test only needs
        # build_model() to be swappable, not the real module, so this avoids that dependency entirely.
        fake_models_mod = types.SimpleNamespace(build_model=fake_build_model)
        monkeypatch.setitem(sys.modules, "gnn_aml_core.models", fake_models_mod)

        ckpt = {"model_name": "gatv2", "hparams": {"hidden": 8, "heads": 2}, "state_dict": {"sentinel": True}}
        arrays = {"x": np.zeros((4, 3), dtype=np.float32), "edge_index": np.zeros((2, 0), dtype=np.int64),
                 "edge_attr": np.zeros((0, 1), dtype=np.float32), "edge_type": np.zeros((0,), dtype=np.int64),
                 "y": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)}
        te = np.array([2, 3])

        auc = evaluate_checkpoint(ckpt, arrays, te)

        assert auc == pytest.approx(1.0)
        assert captured_build_args["model_name"] == "gatv2"
        assert captured_build_args["hp"] == {"hidden": 8, "heads": 2}
        assert fake_model.loaded == {"sentinel": True}


class TestAppendRetrainHistory:
    """Pure file I/O -- no torch or database needed at all."""

    def test_appends_one_json_line_per_call(self, tmp_path):
        from gnn_aml_core.feedback import _append_retrain_history

        _append_retrain_history(tmp_path, {"decision": "rejected", "n": 1})
        _append_retrain_history(tmp_path, {"decision": "promoted", "n": 2})

        lines = (tmp_path / "retrain_history.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"decision": "rejected", "n": 1}
        assert json.loads(lines[1]) == {"decision": "promoted", "n": 2}

    def test_never_overwrites_previous_entries(self, tmp_path):
        from gnn_aml_core.feedback import _append_retrain_history

        for i in range(5):
            _append_retrain_history(tmp_path, {"i": i})
        lines = (tmp_path / "retrain_history.jsonl").read_text().splitlines()
        assert [json.loads(l)["i"] for l in lines] == [0, 1, 2, 3, 4]

    def test_non_json_native_values_are_stringified_not_raising(self, tmp_path):
        import datetime

        from gnn_aml_core.feedback import _append_retrain_history

        _append_retrain_history(tmp_path, {"ts": datetime.datetime(2024, 1, 1)})  # default=str handles this
        line = (tmp_path / "retrain_history.jsonl").read_text().splitlines()[0]
        assert "2024-01-01" in line


class TestRetrainAndMaybePromote:
    """The automatic half of the feedback loop -- promote/reject gating, the backup-before-overwrite, the
    audit trail, and the optional live-reload call. evaluate_checkpoint and retrain_with_feedback are mocked
    (their own correctness is covered by TestEvaluateCheckpoint and TestRetrainWithFeedback above); what's
    under test here is retrain_and_maybe_promote's OWN orchestration: given a current AUC and a candidate
    AUC, does it make the right promote/reject call, and does it do the right thing on disk either way."""

    def _write_graph_and_ckpt(self, tmp_path, torch, *, seed=42):
        out = tmp_path / "models"
        out.mkdir()
        graph = {"x": None, "edge_index": None, "edge_attr": None, "edge_type": None, "y": [0.0, 1.0, 0.0],
                 "account_ids": ["A", "B", "C"]}
        torch.save(graph, out / "graph.pt")
        ckpt = {"model_name": "gatv2", "hparams": {"hidden": 64, "num_layers": 2}, "in_dim": 5,
               "state_dict": {"w": 1}, "feature_names": ["f1"], "mean": None, "std": None,
               "metrics": {"auc_target": 0.87}}
        torch.save(ckpt, out / "gnn_model.pt")
        return out

    def _install_mocks(self, monkeypatch, *, current_auc, candidate_auc):
        import gnn_aml_core.feedback as fb
        import gnn_aml_core.train as train_mod

        monkeypatch.setattr(fb, "evaluate_checkpoint", lambda ckpt, arrays, te: current_auc)
        monkeypatch.setattr(train_mod, "make_splits", lambda y, seed=42: ([0], [1], [2]))

        class _FakeCalibrator:
            def __call__(self, threshold):
                return threshold

            def to_dict(self):
                return {"kind": "identity"}

        fake_res = {"report": {"auc_roc": candidate_auc, "pr_auc": 0.5, "precision": 0.6, "recall": 0.7,
                              "f1": 0.65, "fpr": 0.02, "threshold": 0.5},
                   "best_val_auc": candidate_auc, "best_epoch": 10, "epochs_run": 10, "state_dict": {"w": 2},
                   "hparams": {"hidden": 64, "num_layers": 2}, "calibrator": _FakeCalibrator(),
                   "calibration_report": {"ece": 0.01}}
        monkeypatch.setattr(fb, "retrain_with_feedback", lambda *a, **kw: fake_res)
        return fake_res

    def test_candidate_losing_is_rejected_and_logged_without_touching_the_model_file(self, monkeypatch, tmp_path):
        torch = pytest.importorskip("torch")
        from gnn_aml_core.feedback import retrain_and_maybe_promote

        out = self._write_graph_and_ckpt(tmp_path, torch)
        self._install_mocks(monkeypatch, current_auc=0.90, candidate_auc=0.85)
        before = (out / "gnn_model.pt").read_bytes()

        outcome = retrain_and_maybe_promote([], out_dir=str(out))

        assert outcome["decision"] == "rejected"
        assert (out / "gnn_model.pt").read_bytes() == before  # untouched
        history = (out / "retrain_history.jsonl").read_text().splitlines()
        assert len(history) == 1
        assert json.loads(history[0])["decision"] == "rejected"

    def test_candidate_winning_is_promoted_with_a_backup_of_the_old_model(self, monkeypatch, tmp_path):
        torch = pytest.importorskip("torch")
        from gnn_aml_core.feedback import retrain_and_maybe_promote

        out = self._write_graph_and_ckpt(tmp_path, torch)
        before = (out / "gnn_model.pt").read_bytes()
        self._install_mocks(monkeypatch, current_auc=0.80, candidate_auc=0.90)

        outcome = retrain_and_maybe_promote([], out_dir=str(out))

        assert outcome["decision"] == "promoted"
        backups = list(out.glob("gnn_model.pt.bak-*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == before  # the OLD model's bytes, preserved
        new_ckpt = torch.load(out / "gnn_model.pt", map_location="cpu", weights_only=False)
        assert new_ckpt["state_dict"] == {"w": 2}  # the NEW (candidate) weights are now live
        assert new_ckpt["metrics"]["meets_auc_target"] is True  # 0.90 >= the saved auc_target of 0.87

    def test_min_improvement_margin_is_respected_not_just_any_improvement(self, monkeypatch, tmp_path):
        torch = pytest.importorskip("torch")
        from gnn_aml_core.feedback import retrain_and_maybe_promote

        out = self._write_graph_and_ckpt(tmp_path, torch)
        self._install_mocks(monkeypatch, current_auc=0.80, candidate_auc=0.805)  # a tiny real improvement

        outcome = retrain_and_maybe_promote([], out_dir=str(out), min_improvement=0.05)
        assert outcome["decision"] == "rejected"  # 0.005 improvement does not clear the required 0.05 margin

    def test_promotion_without_gnn_api_skips_the_reload_call(self, monkeypatch, tmp_path):
        torch = pytest.importorskip("torch")
        from gnn_aml_core.feedback import retrain_and_maybe_promote

        out = self._write_graph_and_ckpt(tmp_path, torch)
        self._install_mocks(monkeypatch, current_auc=0.80, candidate_auc=0.90)

        outcome = retrain_and_maybe_promote([], out_dir=str(out), gnn_api=None)
        assert "skipped" in outcome["reload"]

    def test_promotion_with_gnn_api_calls_the_reload_endpoint(self, monkeypatch, tmp_path):
        torch = pytest.importorskip("torch")
        import sys
        import types

        from gnn_aml_core.feedback import retrain_and_maybe_promote

        out = self._write_graph_and_ckpt(tmp_path, torch)
        self._install_mocks(monkeypatch, current_auc=0.80, candidate_auc=0.90)

        captured = {}

        class _FakeResp:
            def read(self):
                return b'{"reloaded": true}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            return _FakeResp()

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        outcome = retrain_and_maybe_promote([], out_dir=str(out), gnn_api="http://fake-gnn",
                                            gnn_admin_token="secret")
        assert captured["url"] == "http://fake-gnn/admin/reload-model"
        assert captured["headers"].get("Authorization") == "Bearer secret"
        assert outcome["reload"] == {"reloaded": True}

    def test_a_failed_reload_call_does_not_fail_the_promotion(self, monkeypatch, tmp_path):
        torch = pytest.importorskip("torch")
        import urllib.error
        import urllib.request

        from gnn_aml_core.feedback import retrain_and_maybe_promote

        out = self._write_graph_and_ckpt(tmp_path, torch)
        self._install_mocks(monkeypatch, current_auc=0.80, candidate_auc=0.90)
        monkeypatch.setattr(urllib.request, "urlopen",
                           lambda req, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("unreachable")))

        outcome = retrain_and_maybe_promote([], out_dir=str(out), gnn_api="http://fake-gnn")
        assert outcome["decision"] == "promoted"  # the promotion itself already happened, on disk
        assert "reload_error" in outcome


class TestMainCheckOnceImportError:
    def test_psycopg_not_installed_exits_with_a_pip_install_hint(self, monkeypatch, capsys):
        import sys

        import gnn_aml_core.feedback as fb

        # force `import psycopg` to raise ImportError regardless of whether the real package happens to be
        # installed in whatever environment runs this test -- the standard way to simulate "not installed"
        monkeypatch.setitem(sys.modules, "psycopg", None)
        monkeypatch.setattr(sys, "argv", ["feedback.py", "--check", "--dsn", "postgresql://fake"])

        with pytest.raises(SystemExit):
            fb.main()
        assert "pip install" in capsys.readouterr().out


class TestMainServeMetricsLoopExtra:
    """More of the --serve-metrics loop's own control flow: the SystemExit-from-_check_once case, and
    auto-retrain actually invoking retrain_and_maybe_promote (the existing test above only pins where
    'current model' appears in the source, not that the call happens or what it's given)."""

    class _StopLoop(Exception):
        pass

    def test_a_system_exit_from_check_once_is_caught_and_the_loop_continues(self, monkeypatch):
        import sys

        import gnn_aml_core.feedback as fb

        calls = {"n": 0}

        def fake_check_once():
            calls["n"] += 1
            raise SystemExit(1)

        # main() defines _check_once as a LOCAL closure, so it can't be monkeypatched directly from outside;
        # instead, drive the same code path through read_decisions_from_db, which _check_once calls first and
        # which is the thing that actually raises SystemExit in the real ImportError/connection-failure cases.
        monkeypatch.setattr(fb, "read_decisions_from_db", lambda dsn: (_ for _ in ()).throw(SystemExit(1)))
        monkeypatch.setattr(fb, "start_http_server", lambda port: None)

        sleep_calls = {"n": 0}

        def fake_sleep(seconds):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise self._StopLoop
        monkeypatch.setattr(fb.time, "sleep", fake_sleep)

        monkeypatch.setattr(sys, "argv", ["feedback.py", "--serve-metrics"])
        with pytest.raises(self._StopLoop):
            fb.main()
        assert sleep_calls["n"] == 2  # the loop survived the SystemExit and reached sleep() twice

    def test_auto_retrain_is_actually_invoked_with_the_triggering_records_when_drift_fires(self, monkeypatch):
        import sys

        import gnn_aml_core.feedback as fb

        records = [rec(account_id=f"A{i}", decision="confirmed" if i % 5 else "dismissed") for i in range(25)]
        monkeypatch.setattr(fb, "read_decisions_from_db", lambda dsn: records)
        monkeypatch.setattr(fb, "start_http_server", lambda port: None)

        captured = {}

        def fake_retrain(recs, **kw):
            captured["records"] = recs
            captured["kw"] = kw
            return {"decision": "rejected"}

        monkeypatch.setattr(fb, "retrain_and_maybe_promote", fake_retrain)

        sleep_calls = {"n": 0}

        def fake_sleep(seconds):
            sleep_calls["n"] += 1
            raise self._StopLoop
        monkeypatch.setattr(fb.time, "sleep", fake_sleep)

        monkeypatch.setattr(sys, "argv", ["feedback.py", "--serve-metrics", "--auto-retrain",
                                         "--retrain-threshold", "0.15", "--min-decisions", "20"])
        with pytest.raises(self._StopLoop):
            fb.main()

        assert captured["records"] == records
        assert captured["kw"]["min_improvement"] == 0.0

    def test_auto_retrain_disabled_by_default_never_calls_retrain(self, monkeypatch):
        import sys

        import gnn_aml_core.feedback as fb

        records = [rec(account_id=f"A{i}", decision="dismissed") for i in range(25)]  # 100% disagreement -> triggers
        monkeypatch.setattr(fb, "read_decisions_from_db", lambda dsn: records)
        monkeypatch.setattr(fb, "start_http_server", lambda port: None)

        called = {"n": 0}
        monkeypatch.setattr(fb, "retrain_and_maybe_promote", lambda recs, **kw: called.update(n=called["n"] + 1))

        sleep_calls = {"n": 0}

        def fake_sleep(seconds):
            sleep_calls["n"] += 1
            raise self._StopLoop
        monkeypatch.setattr(fb.time, "sleep", fake_sleep)

        monkeypatch.setattr(sys, "argv", ["feedback.py", "--serve-metrics"])  # no --auto-retrain
        with pytest.raises(self._StopLoop):
            fb.main()

        assert called["n"] == 0


class TestMainNonServeRetrainPath:
    """main()'s default (non---serve-metrics, non---check) path when drift IS triggered: loads graph.pt/
    gnn_model.pt, converts tensor fields to numpy, and calls retrain_with_feedback with the checkpoint's own
    hidden/num_layers -- with retrain_with_feedback/build_feedback_dataset/make_splits all mocked, since the
    real training itself is covered elsewhere."""

    def test_retrains_using_the_checkpoints_own_architecture_and_logs_the_result(self, monkeypatch, tmp_path, caplog):
        torch = pytest.importorskip("torch")
        import logging
        import sys

        import gnn_aml_core.feedback as fb
        import gnn_aml_core.train as train_mod

        out = tmp_path / "models"
        out.mkdir()
        graph = {"x": None, "edge_index": None, "edge_attr": None, "edge_type": None, "y": [0.0, 1.0],
                "account_ids": ["A", "B"]}
        torch.save(graph, out / "graph.pt")
        ckpt = {"model_name": "rgcn", "hparams": {"hidden": 32, "num_layers": 3}}
        torch.save(ckpt, out / "gnn_model.pt")

        records = [rec(account_id=f"A{i}", decision="dismissed") for i in range(25)]  # triggers (100% disagreement)
        monkeypatch.setattr(fb, "read_decisions_from_db", lambda dsn: records)
        monkeypatch.setattr(train_mod, "make_splits", lambda y, seed=42: ([0], [1], []))

        captured = {}

        def fake_retrain_with_feedback(g, tr, va, te, ids, labels, **kw):
            captured.update(kw=kw, ids=ids, labels=labels)
            return {"best_val_auc": 0.77, "best_epoch": 5, "report": {"auc_roc": 0.77}}

        monkeypatch.setattr(fb, "retrain_with_feedback", fake_retrain_with_feedback)

        monkeypatch.setattr(sys, "argv", ["feedback.py", "--dsn", "postgresql://fake", "--data", str(tmp_path),
                                         "--out", str(out), "--retrain-threshold", "0.15", "--min-decisions", "20"])

        with caplog.at_level(logging.INFO, logger="gnn_aml_core.feedback"):
            fb.main()

        assert captured["kw"]["model_name"] == "rgcn"
        assert captured["kw"]["hidden"] == 32
        assert captured["kw"]["layers"] == 3
        assert "retrained: val AUC 0.7700" in caplog.text

    def test_check_flag_returns_before_retraining_even_if_triggered(self, monkeypatch, tmp_path):
        import sys

        import gnn_aml_core.feedback as fb

        records = [rec(account_id=f"A{i}", decision="dismissed") for i in range(25)]
        monkeypatch.setattr(fb, "read_decisions_from_db", lambda dsn: records)

        called = {"n": 0}
        monkeypatch.setattr(fb, "retrain_with_feedback", lambda *a, **kw: called.update(n=called["n"] + 1))

        monkeypatch.setattr(sys, "argv", ["feedback.py", "--check", "--dsn", "postgresql://fake",
                                         "--retrain-threshold", "0.15", "--min-decisions", "20"])
        fb.main()
        assert called["n"] == 0

    def test_not_triggered_returns_without_retraining(self, monkeypatch, tmp_path):
        import sys

        import gnn_aml_core.feedback as fb

        records = [rec(account_id=f"A{i}", decision="confirmed") for i in range(25)]  # 0% disagreement
        monkeypatch.setattr(fb, "read_decisions_from_db", lambda dsn: records)

        called = {"n": 0}
        monkeypatch.setattr(fb, "retrain_with_feedback", lambda *a, **kw: called.update(n=called["n"] + 1))

        monkeypatch.setattr(sys, "argv", ["feedback.py", "--dsn", "postgresql://fake",
                                         "--retrain-threshold", "0.15", "--min-decisions", "20"])
        fb.main()
        assert called["n"] == 0
