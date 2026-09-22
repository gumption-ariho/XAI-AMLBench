"""Shared pytest fixtures. Keep this file free of heavy imports (torch) so tests that don't need it stay fast
and so the whole suite can still run (bar the torch-marked tests) on a machine without PyTorch installed."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: needs PyTorch (skipped automatically if it is not installed)")


@pytest.fixture(scope="session")
def small_graph():
    """One small synthetic graph, reused (read-only) across tests that don't need to control its parameters."""
    from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig
    return AMLGraphGenerator(GeneratorConfig(n_accounts=800, n_background_tx=3000, n_patterns_per_typology=3, seed=7)).generate()


@pytest.fixture(scope="session")
def has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False
