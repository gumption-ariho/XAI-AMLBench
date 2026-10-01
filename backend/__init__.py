"""backend -- the ProjectXY core API (container: backend).

Deliberately no re-exports here (unlike aml_synth/gnn_aml_core/xai_explainer's __init__.py files): backend.main
reads DATABASE_URL from the environment at import time, so importing it unconditionally from here would force
that requirement onto anything that imports the backend package at all, including backend.manage_keys callers
who may not need main's routes.
"""
