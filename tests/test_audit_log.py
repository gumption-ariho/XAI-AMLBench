"""Tests for xai_explainer.audit_log. Pure stdlib (hashlib, json, threading) -- no PyTorch, no external
services, so these always run. Postgres mirroring is tested only for graceful degradation (a failed or absent
connection must never break the file-based log), since a live database is not assumed in this test suite."""
import json
import threading

import pytest

from xai_explainer.audit_log import GENESIS_HASH, AuditLog, _entry_hash


@pytest.fixture
def logfile(tmp_path):
    return str(tmp_path / "audit.jsonl")


class TestAppend:
    def test_first_entry_has_seq_zero_and_genesis_prev_hash(self, logfile):
        entry = AuditLog(logfile).append({"x": 1})
        assert entry["seq"] == 0
        assert entry["prev_hash"] == GENESIS_HASH

    def test_returns_the_full_stored_entry(self, logfile):
        entry = AuditLog(logfile).append({"x": 1})
        assert set(entry) == {"seq", "ts", "record", "prev_hash", "hash", "retain_until"}
        assert entry["record"] == {"x": 1}

    def test_sequence_numbers_increment(self, logfile):
        log = AuditLog(logfile)
        seqs = [log.append({"i": i})["seq"] for i in range(5)]
        assert seqs == [0, 1, 2, 3, 4]

    def test_each_entry_chains_to_the_previous_hash(self, logfile):
        log = AuditLog(logfile)
        e0 = log.append({"i": 0})
        e1 = log.append({"i": 1})
        assert e1["prev_hash"] == e0["hash"]

    def test_creates_parent_directories(self, tmp_path):
        p = str(tmp_path / "nested" / "deep" / "audit.jsonl")
        AuditLog(p).append({"x": 1})
        assert (tmp_path / "nested" / "deep" / "audit.jsonl").exists()

    def test_retain_until_defaults_to_five_years_out(self, logfile):
        entry = AuditLog(logfile).append({"x": 1})
        years = (entry["retain_until"] - entry["ts"]) / (365.25 * 86400)
        assert years == pytest.approx(5.0, abs=0.01)

    def test_retention_years_is_configurable(self, logfile):
        entry = AuditLog(logfile, retention_years=1.0).append({"x": 1})
        years = (entry["retain_until"] - entry["ts"]) / (365.25 * 86400)
        assert years == pytest.approx(1.0, abs=0.01)

    def test_concurrent_appends_from_multiple_threads_produce_a_valid_chain(self, logfile):
        # A backend serving concurrent requests could call append() from more than one thread at once; the
        # internal lock must serialise sequence numbers and hash chaining correctly regardless.
        log = AuditLog(logfile)

        def worker(n):
            for i in range(n):
                log.append({"thread_local_i": i})

        threads = [threading.Thread(target=worker, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(log) == 50
        assert log.verify()
        seqs = sorted(e["seq"] for e in log.read_all())
        assert seqs == list(range(50))


class TestReadAll:
    def test_empty_log_yields_nothing(self, logfile):
        assert list(AuditLog(logfile).read_all()) == []

    def test_yields_entries_in_append_order(self, logfile):
        log = AuditLog(logfile)
        for i in range(4):
            log.append({"i": i})
        assert [e["record"]["i"] for e in log.read_all()] == [0, 1, 2, 3]

    def test_len_matches_number_of_entries(self, logfile):
        log = AuditLog(logfile)
        for i in range(7):
            log.append({"i": i})
        assert len(log) == 7


class TestReopening:
    def test_resumes_the_chain_from_an_existing_file(self, logfile):
        log1 = AuditLog(logfile)
        for i in range(3):
            log1.append({"i": i})
        log2 = AuditLog(logfile)  # a fresh instance pointed at the same file
        entry = log2.append({"i": 3})
        assert entry["seq"] == 3
        assert entry["prev_hash"] != GENESIS_HASH

    def test_reopened_log_still_verifies(self, logfile):
        AuditLog(logfile).append({"x": 1})
        assert AuditLog(logfile).verify()


class TestVerify:
    def test_empty_or_missing_file_verifies_true(self, logfile):
        assert AuditLog(logfile).verify()  # nothing appended yet, no file exists

    def test_untampered_log_verifies_true(self, logfile):
        log = AuditLog(logfile)
        for i in range(5):
            log.append({"n": i})
        assert log.verify()

    def test_detects_a_value_changed_in_an_old_entry(self, logfile):
        # This is the exact scenario check_brief_v1.py exercises: a hand-edit of the raw file that changes a
        # value without touching the (now stale) stored hash for that line.
        log = AuditLog(logfile)
        for i in range(5):
            log.append({"n": i})
        from pathlib import Path
        raw = Path(logfile).read_text()
        Path(logfile).write_text(raw.replace('"n": 2', '"n": 9', 1))
        assert not AuditLog(logfile).verify()

    def test_detects_a_deleted_entry(self, logfile):
        from pathlib import Path
        log = AuditLog(logfile)
        for i in range(5):
            log.append({"i": i})
        lines = Path(logfile).read_text().splitlines()
        del lines[2]
        Path(logfile).write_text("\n".join(lines) + "\n")
        assert not AuditLog(logfile).verify()

    def test_detects_reordered_entries(self, logfile):
        from pathlib import Path
        log = AuditLog(logfile)
        for i in range(4):
            log.append({"i": i})
        lines = Path(logfile).read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        Path(logfile).write_text("\n".join(lines) + "\n")
        assert not AuditLog(logfile).verify()

    def test_detects_a_forged_hash_via_the_broken_downstream_link(self, logfile):
        # Even if an attacker recomputes a consistent hash for the entry they edited, the NEXT entry's
        # prev_hash was computed from the entry's ORIGINAL hash, so the chain still breaks one step later.
        from pathlib import Path
        log = AuditLog(logfile)
        for i in range(3):
            log.append({"n": i})
        lines = Path(logfile).read_text().splitlines()
        entry = json.loads(lines[1])
        entry["record"]["n"] = 999
        entry["hash"] = _entry_hash(entry["seq"], entry["ts"], entry["record"], entry["prev_hash"])
        lines[1] = json.dumps(entry, sort_keys=True)
        Path(logfile).write_text("\n".join(lines) + "\n")
        assert not AuditLog(logfile).verify()

    def test_detects_garbage_appended_to_the_file(self, logfile):
        from pathlib import Path
        log = AuditLog(logfile)
        log.append({"x": 1})
        with open(logfile, "a") as f:
            f.write("not valid json at all\n")
        assert not AuditLog(logfile).verify()


class TestPostgresMirrorDegradesGracefully:
    def test_unreachable_database_does_not_prevent_file_writes(self, logfile):
        log = AuditLog(logfile, database_url="postgresql://nouser:nopass@localhost:1/nodb")
        entry = log.append({"x": 1})
        assert "hash" in entry
        assert AuditLog(logfile).verify()

    def test_unreachable_database_disables_further_attempts(self, logfile):
        log = AuditLog(logfile, database_url="postgresql://nouser:nopass@localhost:1/nodb")
        log.append({"x": 1})
        assert log._database_url is None  # the constructor's connectivity check already failed and gave up
