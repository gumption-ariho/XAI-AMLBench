"""aml_synth -- synthetic multi-hop transaction networks with 5 laundering typologies.

Usage: python -m aml_synth.graph_generator --help
"""
from aml_synth.graph_generator import AMLGraphGenerator, GeneratorConfig, SyntheticGraph, TYPOLOGIES

__all__ = ["AMLGraphGenerator", "GeneratorConfig", "SyntheticGraph", "TYPOLOGIES"]
__version__ = "0.1.0"
