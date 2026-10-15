"""Tests for gnn_aml_core.anomaly: Model 2, the graph-autoencoder anomaly detector.

gnn_aml_core/anomaly.py does `from prometheus_client import Counter, Gauge, start_http_server` at module
level (unlike gnn_aml_core.feedback's own psycopg/torch imports, which are all deferred into individual
functions), so importing this module -- and therefore everything in this file -- needs prometheus_client
installed, same as the rest of this suite needs torch and psycopg for the functions that use them. Nothing
here talks to a real database, a real Kafka broker, a real GNN API, or does real GPU training: psycopg,
httpx, and torch are all stubbed via sys.modules the same way tests/test_feedback.py's
TestMainErrorMessages already does for psycopg, so these tests exercise anomaly.py's own logic (what goes
into the autoencoder's training population, how the anomaly threshold is computed, which accounts get a new
alert row, how a confirmed Model-2 case reaches Model 1, and that a bad cycle never kills the --serve-metrics
loop) without needing any of the real services this module talks to in production.
"""
import sys
import types

import numpy as np
import pytest

import gnn_aml_core.anomaly as an
from gnn_aml_core.anomaly import AnomalyHit, score_population


# ---------------------------------------------------------------------------------------------------------
# score_population: pure numpy, no heavy deps -- the actual statistics this module relies on to decide what
# "unusual" means, so each property below is pinned on its own rather than only through one aggregate case.
# ---------------------------------------------------------------------------------------------------------
class TestScorePopulation:
    def test_too_few_normal_accounts_returns_no_hits_and_infinite_threshold(self):
        # fewer than 2 normal accounts -- median/MAD are meaningless, so this must not raise or guess
        recon_error = np.array([0.1, 0.2, 0.3])
        normal_mask = np.array([True, False, False])
        hits, threshold = score_population(recon_error, ["A", "B", "C"], normal_mask)
        assert hits == [] and threshold == float("inf")

    def test_flagged_accounts_are_never_candidates_even_with_extreme_error(self):
        # Model 1 already flagged B -- Model 2 must never re-alert on it, however anomalous its reconstruction
        recon_error = np.array([0.1, 50.0, 0.12, 0.11])
        normal_mask = np.array([True, False, True, True])
        hits, _ = score_population(recon_error, ["A", "B", "C", "D"], normal_mask)
        assert all(h.account_id != "B" for h in hits)

    def test_a_normal_account_far_above_the_pack_is_flagged(self):
        normal_errors = [0.10, 0.11, 0.09, 0.10, 0.12, 0.11, 0.10, 5.0]  # last one is the outlier
        recon_error = np.array(normal_errors)
        normal_mask = np.array([True] * 8)
        ids = [f"N{i}" for i in range(8)]
        hits, threshold = score_population(recon_error, ids, normal_mask)
        assert any(h.account_id == "N7" for h in hits)
        assert threshold < 5.0

    def test_a_uniformly_tight_normal_population_flags_nobody(self):
        recon_error = np.array([0.10, 0.101, 0.099, 0.1005, 0.0995])
        normal_mask = np.array([True] * 5)
        hits, _ = score_population(recon_error, [f"N{i}" for i in range(5)], normal_mask)
        assert hits == []

    def test_outliers_do_not_inflate_their_own_detection_threshold(self):
        # the real bug this design is pinned against (see the module docstring): with mean/stdev, a handful
        # of extreme values would drag the threshold UP past themselves and hide them. Median/MAD must not.
        tight = [0.10] * 20
        extreme = [9.0, 9.5, 10.0]  # 3 extreme outliers mixed into an otherwise tight population
        recon_error = np.array(tight + extreme)
        normal_mask = np.array([True] * len(recon_error))
        ids = [f"N{i}" for i in range(len(recon_error))]
        hits, threshold = score_population(recon_error, ids, normal_mask)
        hit_ids = {h.account_id for h in hits}
        assert {"N20", "N21", "N22"} <= hit_ids  # all 3 extreme accounts are still caught
        assert threshold < 9.0

    def test_returned_hit_carries_the_real_score_and_threshold_used(self):
        recon_error = np.array([0.1] * 9 + [10.0])
        normal_mask = np.array([True] * 10)
        hits, threshold = score_population(recon_error, [f"N{i}" for i in range(10)], normal_mask)
        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, AnomalyHit)
        assert hit.account_id == "N9"
        assert hit.score == pytest.approx(10.0)
        assert hit.threshold == pytest.approx(threshold)

    def test_zero_variance_normal_population_uses_the_median_itself_as_threshold(self):
        # robust_std == 0 (every normal account reconstructs identically) -- the k_std*robust_std term must
        # not be used as a multiplier on zero (which would also give 0 and flag the entire population)
        recon_error = np.array([0.5, 0.5, 0.5, 0.5, 5.0])
        normal_mask = np.array([True, True, True, True, False])  # the outlier is Model-1-flagged, excluded
        hits, threshold = score_population(recon_error, ["A", "B", "C", "D", "E"], normal_mask)
        assert threshold == pytest.approx(0.5)
        assert hits == []  # nothing strictly exceeds the threshold


class TestUpsertAlert:
    """_upsert_alert takes a plain `conn` -- no psycopg import needed to test it, a fake with the same
    .execute()/.fetchone() shape is enough."""

    class _FakeCursor:
        def __init__(self, fetchone_result):
            self._fetchone_result = fetchone_result

        def fetchone(self):
            return self._fetchone_result

    class _FakeConn:
        def __init__(self, existing_alert=None):
            self.existing_alert = existing_alert
            self.executed = []

        def execute(self, sql, params=None):
            self.executed.append((sql, params))
            if sql.strip().startswith("SELECT"):
                return TestUpsertAlert._FakeCursor(self.existing_alert)
            return TestUpsertAlert._FakeCursor(None)

    def test_creates_a_new_alert_when_none_is_open(self):
        conn = self._FakeConn(existing_alert=None)
        hit = AnomalyHit(account_id="A1", score=5.0, threshold=1.0)
        created = an._upsert_alert(conn, hit)
        assert created is True
        insert_calls = [c for c in conn.executed if c[0].strip().startswith("INSERT")]
        assert len(insert_calls) == 1
        assert insert_calls[0][1] == ("A1", 5.0, 1.0)

    def test_does_not_duplicate_an_already_open_alert(self):
        conn = self._FakeConn(existing_alert={"id": 7})
        hit = AnomalyHit(account_id="A1", score=5.0, threshold=1.0)
        created = an._upsert_alert(conn, hit)
        assert created is False
        insert_calls = [c for c in conn.executed if c[0].strip().startswith("INSERT")]
        assert insert_calls == []


# ---------------------------------------------------------------------------------------------------------
# anomaly_pass / sync_confirmed_to_model1: these do real torch, psycopg, and httpx work in production.
# Stubbed via sys.modules (same technique as test_feedback.py's TestMainErrorMessages) so the test exercises
# anomaly.py's own control flow -- what it does with Model 1's predictions, what it persists, what it
# excludes -- without any of those services existing.
# ---------------------------------------------------------------------------------------------------------
class TestAnomalyPass:
    def _install_fake_deps(self, monkeypatch, *, predict_results, upsert_created=None):
        fake_torch = types.SimpleNamespace()
        saved = {}

        def fake_load(path, map_location=None, weights_only=None):
            # graph.pt is always loaded; anomaly_autoencoder.pt is only loaded if it "exists" -- the test
            # controls that through monkeypatching Path.exists below, so this stub just needs to hand back
            # whatever the test wants for either call, in call order.
            return saved["load_queue"].pop(0)

        fake_torch.load = fake_load
        fake_torch.save = lambda obj, path: saved.setdefault("saved", []).append((obj, path))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        fake_httpx_response = types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"results": predict_results},
        )
        fake_httpx = types.SimpleNamespace(post=lambda url, json=None, timeout=None: fake_httpx_response)
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

        upserted = []

        class _FakeConn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake_psycopg = types.SimpleNamespace(connect=lambda dsn, row_factory=None: _FakeConn())
        fake_rows_mod = types.SimpleNamespace(dict_row=object())
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
        monkeypatch.setitem(sys.modules, "psycopg.rows", fake_rows_mod)

        monkeypatch.setattr(an, "_upsert_alert", lambda conn, hit: upserted.append(hit.account_id) or True)
        monkeypatch.setattr(an, "_db_url", lambda: "postgresql://fake")

        return saved, upserted

    def test_accounts_model1_already_flags_are_excluded_from_the_training_population(self, monkeypatch, tmp_path):
        account_ids = ["A", "B", "C"]
        fake_graph = {"x": np.zeros((3, 2), dtype=np.float32), "edge_index": np.zeros((2, 0), dtype=np.int64),
                     "edge_attr": np.zeros((0, 1), dtype=np.float32), "account_ids": account_ids}
        saved, _ = self._install_fake_deps(
            monkeypatch,
            predict_results=[{"account_id": "A", "flagged": False}, {"account_id": "B", "flagged": True},
                             {"account_id": "C", "flagged": False}],
        )
        saved["load_queue"] = [fake_graph]  # no checkpoint on disk yet -> only graph.pt is loaded

        captured_normal_idx = {}

        def fake_train_autoencoder(arrays, normal_idx, **kw):
            captured_normal_idx["idx"] = list(normal_idx)
            return {"state_dict": {}, "recon_error": np.array([0.1, 0.1, 0.1]), "train_loss": 0.01,
                    "hparams": {"hidden": 64, "heads": 4, "layers": 2, "dropout": 0.2}}

        monkeypatch.setattr(an, "train_autoencoder", fake_train_autoencoder)
        monkeypatch.setattr(an, "score_population", lambda *a, **kw: ([], float("inf")))

        out_dir = tmp_path / "models"
        out_dir.mkdir()
        an.anomaly_pass(data_dir=str(tmp_path), out_dir=str(out_dir), gnn_api="http://fake/gnn")

        # B (index 1) is flagged by Model 1 -- it must never be in the autoencoder's own training population
        assert 1 not in captured_normal_idx["idx"]
        assert set(captured_normal_idx["idx"]) == {0, 2}

    def test_new_alerts_are_raised_only_for_hits_and_counted_correctly(self, monkeypatch, tmp_path):
        # 7 tightly-clustered normal accounts plus 1 real outlier -- 2 normal accounts isn't enough data for
        # score_population's own median/MAD statistics to single out an outlier at the default k_std=3.0
        # (confirmed directly: with just [0.1, 9.0], the outlier itself drags the threshold past 24, so
        # nothing is ever flagged -- that's score_population's own guard against a too-small sample, not a
        # bug in anomaly_pass, so this test gives it a population actually large enough to make the point).
        account_ids = [f"N{i}" for i in range(7)] + ["OUTLIER"]
        n = len(account_ids)
        fake_graph = {"x": np.zeros((n, 2), dtype=np.float32), "edge_index": np.zeros((2, 0), dtype=np.int64),
                     "edge_attr": np.zeros((0, 1), dtype=np.float32), "account_ids": account_ids}
        saved, upserted = self._install_fake_deps(
            monkeypatch,
            predict_results=[{"account_id": aid, "flagged": False} for aid in account_ids],
        )
        saved["load_queue"] = [fake_graph]

        recon_error = np.array([0.10, 0.11, 0.09, 0.10, 0.12, 0.11, 0.10, 9.0])
        monkeypatch.setattr(an, "train_autoencoder", lambda arrays, normal_idx, **kw: {
            "state_dict": {}, "recon_error": recon_error, "train_loss": 0.02,
            "hparams": {"hidden": 64, "heads": 4, "layers": 2, "dropout": 0.2}})

        out_dir = tmp_path / "models"
        out_dir.mkdir()
        result = an.anomaly_pass(data_dir=str(tmp_path), out_dir=str(out_dir), gnn_api="http://fake/gnn")

        assert result["scored"] == n
        assert result["new_alerts"] == len(upserted)
        assert upserted == ["OUTLIER"]  # exactly the real outlier made it through to _upsert_alert

    def test_existing_checkpoint_is_warm_started_not_discarded(self, monkeypatch, tmp_path):
        account_ids = ["A", "B"]
        fake_graph = {"x": np.zeros((2, 2), dtype=np.float32), "edge_index": np.zeros((2, 0), dtype=np.int64),
                     "edge_attr": np.zeros((0, 1), dtype=np.float32), "account_ids": account_ids}
        prev_state = {"some": "weights"}
        fake_ckpt = {"state_dict": prev_state, "hparams": {"hidden": 32, "heads": 2, "layers": 1}}
        saved, _ = self._install_fake_deps(
            monkeypatch,
            predict_results=[{"account_id": "A", "flagged": False}, {"account_id": "B", "flagged": False}],
        )
        saved["load_queue"] = [fake_graph, fake_ckpt]

        out_dir = tmp_path / "models"
        out_dir.mkdir()
        (out_dir / "anomaly_autoencoder.pt").write_bytes(b"not actually read -- torch.load is stubbed")

        captured = {}

        def fake_train_autoencoder(arrays, normal_idx, *, hidden, heads, layers, init_state_dict=None, **kw):
            captured["init_state_dict"] = init_state_dict
            captured["hidden"] = hidden
            return {"state_dict": {}, "recon_error": np.array([0.1, 0.1]), "train_loss": 0.01,
                    "hparams": {"hidden": hidden, "heads": heads, "layers": layers, "dropout": 0.2}}

        monkeypatch.setattr(an, "train_autoencoder", fake_train_autoencoder)

        an.anomaly_pass(data_dir=str(tmp_path), out_dir=str(out_dir), gnn_api="http://fake/gnn")

        assert captured["init_state_dict"] is prev_state  # warm-started from the checkpoint, not None
        assert captured["hidden"] == 32  # the checkpoint's saved hparams override the call's defaults


class TestSyncConfirmedToModel1:
    def test_nothing_to_sync_returns_empty_list_without_calling_retrain(self, monkeypatch):
        fake_cursor_result = types.SimpleNamespace(fetchall=lambda: [])

        class _FakeConn:
            def execute(self, sql, params=None):
                return fake_cursor_result

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake_psycopg = types.SimpleNamespace(connect=lambda dsn, row_factory=None: _FakeConn())
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
        monkeypatch.setitem(sys.modules, "psycopg.rows", types.SimpleNamespace(dict_row=object()))
        monkeypatch.setattr(an, "_db_url", lambda: "postgresql://fake")

        called = {"n": 0}
        fake_feedback_mod = types.SimpleNamespace(
            DecisionRecord=lambda **kw: kw,
            retrain_and_maybe_promote=lambda *a, **kw: called.update(n=called["n"] + 1),
        )
        monkeypatch.setitem(sys.modules, "gnn_aml_core.feedback", fake_feedback_mod)

        result = an.sync_confirmed_to_model1(gnn_api="http://fake")
        assert result == []
        assert called["n"] == 0  # retrain_and_maybe_promote must never be called with nothing to teach

    def test_confirmed_rows_are_built_into_decision_records_and_marked_trained(self, monkeypatch):
        rows = [{"id": 1, "account_id": "A1", "anomaly_score": 9.0, "threshold": 1.0},
               {"id": 2, "account_id": "A2", "anomaly_score": 8.0, "threshold": 1.0}]
        executed = []

        class _FakeConn:
            def __init__(self):
                self._select_done = False

            def execute(self, sql, params=None):
                executed.append((sql, params))
                if sql.strip().startswith("SELECT"):
                    return types.SimpleNamespace(fetchall=lambda: rows)
                return types.SimpleNamespace(fetchall=lambda: [])

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake_psycopg = types.SimpleNamespace(connect=lambda dsn, row_factory=None: _FakeConn())
        monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
        monkeypatch.setitem(sys.modules, "psycopg.rows", types.SimpleNamespace(dict_row=object()))
        monkeypatch.setattr(an, "_db_url", lambda: "postgresql://fake")

        captured_records = {}

        def fake_retrain(records, **kw):
            captured_records["records"] = records
            return {"promoted": True}

        fake_feedback_mod = types.SimpleNamespace(
            DecisionRecord=lambda **kw: types.SimpleNamespace(**kw),
            retrain_and_maybe_promote=fake_retrain,
        )
        monkeypatch.setitem(sys.modules, "gnn_aml_core.feedback", fake_feedback_mod)

        result = an.sync_confirmed_to_model1(gnn_api="http://fake")

        assert len(result) == 1
        assert result[0]["source"] == "model2_confirmed_anomalies"
        assert result[0]["account_ids"] == ["A1", "A2"]
        assert [r.decision for r in captured_records["records"]] == ["confirmed", "confirmed"]

        # every confirmed row must be marked trained_at -- otherwise the same case gets re-taught forever
        update_calls = [c for c in executed if c[0].strip().startswith("UPDATE")]
        assert len(update_calls) == 2
        assert {c[1][0] for c in update_calls} == {1, 2}


class TestMain:
    def test_check_flag_runs_one_pass_and_does_not_start_the_metrics_server(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["anomaly.py", "--check", "--gnn-api", "http://fake"])
        called = {}
        monkeypatch.setattr(an, "anomaly_pass", lambda **kw: called.update(kw) or {"scored": 0, "hits": 0,
                                                                                   "new_alerts": 0,
                                                                                   "threshold": float("inf"),
                                                                                   "train_loss": 0.0})
        monkeypatch.setattr(an, "start_http_server", lambda port: pytest.fail("--check must not serve metrics"))
        an.main()
        assert called["gnn_api"] == "http://fake"
        assert "scored" in capsys.readouterr().out

    def test_serve_metrics_loop_survives_a_failing_cycle(self, monkeypatch):
        # a real design requirement, mirrored from gnn_aml_core.feedback's own long-lived loop: one bad cycle
        # (DB hiccup, gnn-detection-api briefly unreachable) must log and retry, never take the process down
        monkeypatch.setattr(sys, "argv", ["anomaly.py", "--serve-metrics"])
        monkeypatch.setattr(an, "start_http_server", lambda port: None)

        call_count = {"n": 0}

        def failing_anomaly_pass(**kw):
            call_count["n"] += 1
            raise RuntimeError("gnn-detection-api unreachable")

        class _StopLoop(Exception):
            pass

        sleep_calls = {"n": 0}

        def fake_sleep(seconds):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                # by the 2nd trip through the loop, anomaly_pass must already have been given a 2nd chance
                # to run -- proving the 1st failure was caught and logged, not left to kill the process
                raise _StopLoop

        monkeypatch.setattr(an, "anomaly_pass", failing_anomaly_pass)
        monkeypatch.setattr(an, "sync_confirmed_to_model1", lambda **kw: [])
        monkeypatch.setattr(an.time, "sleep", fake_sleep)

        with pytest.raises(_StopLoop):
            an.main()

        assert call_count["n"] == 2  # ran again after the 1st failure instead of dying
