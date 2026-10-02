"""Tests for node_agent.agent. The module's __main__ guard means importing it is side-effect free (no HTTP
server, no scheduler loop actually starts) -- real network calls inside check_http/check_tcp are mocked here,
since this project's test suite must not depend on any real service actually running.
"""
import socket
import sys
import time
import urllib.error

import pytest

sys.path.insert(0, "node_agent")
import agent


class TestCheckHttp:
    def test_a_successful_response_reports_up_with_a_timing(self, monkeypatch):
        class FakeResponse:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(agent.urllib.request, "urlopen", lambda url, timeout=3: FakeResponse())
        result = agent.check_http("http://fake/health")
        assert result["up"] is True
        assert "ms" in result

    def test_a_connection_error_reports_down_with_the_exception_type(self, monkeypatch):
        def raise_it(url, timeout=3):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(agent.urllib.request, "urlopen", raise_it)
        result = agent.check_http("http://fake/health")
        assert result["up"] is False
        assert result["error"] == "URLError"

    def test_a_4xx_or_5xx_status_reports_down_not_up(self, monkeypatch):
        class FakeResponse:
            status = 503
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(agent.urllib.request, "urlopen", lambda url, timeout=3: FakeResponse())
        result = agent.check_http("http://fake/health")
        assert result["up"] is False


class TestCheckTcp:
    def test_a_successful_connection_reports_up(self, monkeypatch):
        class FakeSocket:
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(agent.socket, "create_connection", lambda addr, timeout=3: FakeSocket())
        result = agent.check_tcp("fake-host", 1234)
        assert result["up"] is True
        assert "ms" in result

    def test_a_refused_connection_reports_down_with_the_exception_type(self, monkeypatch):
        def raise_it(addr, timeout=3):
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(agent.socket, "create_connection", raise_it)
        result = agent.check_tcp("fake-host", 1234)
        assert result["up"] is False
        assert result["error"] == "ConnectionRefusedError"


class TestHealthSweep:
    def test_aggregates_both_http_and_tcp_targets_and_updates_latest(self, monkeypatch):
        monkeypatch.setattr(agent, "check_http", lambda url: {"up": True, "ms": 1.0})
        monkeypatch.setattr(agent, "check_tcp", lambda host, port: {"up": True, "ms": 1.0})
        agent.health_sweep()
        assert agent.LATEST["checked_at"] is not None
        expected_services = set(agent.HTTP_TARGETS) | set(agent.TCP_TARGETS)
        assert set(agent.LATEST["services"]) == expected_services

    def test_a_down_service_is_correctly_identified_as_down(self, monkeypatch):
        def fake_http(url):
            return {"up": False, "error": "URLError"} if "gnn" in url else {"up": True, "ms": 1.0}
        monkeypatch.setattr(agent, "check_http", fake_http)
        monkeypatch.setattr(agent, "check_tcp", lambda host, port: {"up": True, "ms": 1.0})
        agent.health_sweep()
        assert agent.LATEST["services"]["gnn-detection-api"]["up"] is False
        assert agent.LATEST["services"]["backend"]["up"] is True


class TestScheduler:
    def test_every_registers_a_job(self):
        s = agent.Scheduler()
        s.every(10, lambda: None)
        assert len(s.jobs) == 1
        assert s.jobs[0][0] == 10

    def test_multiple_jobs_can_be_registered(self):
        s = agent.Scheduler()
        s.every(10, lambda: None)
        s.every(20, lambda: None)
        assert len(s.jobs) == 2

    def test_a_failing_job_does_not_stop_the_loop_from_retrying(self):
        # the real risk this guards against: one bad periodic job (e.g. a transient network blip) must not
        # silently kill the whole scheduler thread -- _loop catches and logs, then retries after the interval
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("simulated transient failure")

        t = __import__("threading").Thread(target=agent.Scheduler._loop, args=(0.05, flaky), daemon=True)
        t.start()
        time.sleep(0.3)   # several intervals' worth of time
        assert len(calls) >= 2, "the loop should have retried after the first call raised"
