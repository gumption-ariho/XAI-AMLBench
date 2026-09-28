"""Tests for gnn_aml_core.feedback: the drift-detection and feedback-dataset logic. No PyTorch and no database
needed -- everything except read_decisions_from_db operates on plain DecisionRecord objects, so these always
run. retrain_with_feedback is tested with a stand-in fit_gnn rather than a real one, since only its own
label-overwriting and index-repetition logic belongs to this module; the actual training is gnn_aml_core.train's
responsibility and is tested there.
"""
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
