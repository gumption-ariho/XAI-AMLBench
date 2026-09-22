"""xai_explainer -- turns GNNExplainer subgraphs into validated 3-4 sentence SAR narratives."""
from xai_explainer.sar_generator import build_facts, render_fact_sheet, template_narrative, validate_narrative

__all__ = ["build_facts", "render_fact_sheet", "template_narrative", "validate_narrative"]
__version__ = "0.1.0"
