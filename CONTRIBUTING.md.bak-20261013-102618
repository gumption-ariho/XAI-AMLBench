# Contributing to XAI-AMLBench

Thanks for considering a contribution. This is a research benchmark, so correctness and reproducibility matter
more than speed.

## Getting started

```bash
git clone https://github.com/OWNER/xai-amlbench.git
cd xai-amlbench
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # editable install with test, GNN and XAI extras
# CPU-only PyTorch (smaller download, no GPU needed):
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pytest                            # run the test suite
```

## Before opening a pull request

- **Run the tests:** `pytest --cov` and keep coverage at 80% or above for the module you touched. Tests that need
  PyTorch are marked so they skip cleanly (not fail) when it is not installed; do not remove that behaviour.
- **Run the brief scorecard:** `python3 check_brief_v1.py` and make sure you have not turned a PASS into a FAIL.
- **If you touch the generator (`aml_synth`):** re-run the benchmark
  (`python -m gnn_aml_core.benchmark --seeds 5`) and check that simple per-account models (Logistic Regression,
  boosting) do not jump to near-perfect AUC. The whole point of the benchmark is that it is not trivially solvable
  without the graph; a change that makes it easy again is a regression even if all tests still pass.
- **Keep the docstring accurate.** Several modules document their exact behaviour (edge direction, feature
  definitions, validation rules); update the docstring in the same commit as the code.

## What to work on

- Items marked FAIL in `python3 check_brief_v1.py` are the most useful starting points.
- Extending `gnn_aml_core/features.py` toward the project brief's 400+ feature target.
- A real Llama-3 backend for `xai_explainer` (the CPU "template" backend is the validated fallback; do not remove
  it, since it is what makes narratives reproducible without a GPU).

## Code style

Plain, readable Python; type hints where they help; no new dependencies without a good reason (this project runs
on modest hardware, including CPU-only laptops). Please run `python3 -m py_compile` on any file you add before
opening a pull request.

## Reporting issues

Please include: what you ran, what you expected, what happened, and the output of
`python3 check_brief_v1.py` and/or `python3 check_stack_v1.py` if relevant.
