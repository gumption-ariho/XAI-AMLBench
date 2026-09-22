"""gnn_aml_core -- GATv2 / RGCN money-laundering detection (training + FastAPI serving).

Heavy imports (torch, torch_geometric) are NOT pulled in at package import time, so
`import gnn_aml_core` stays cheap even when those are not installed; import from the
submodules directly, e.g. `from gnn_aml_core.models import build_model`.
"""
__version__ = "0.1.0"
