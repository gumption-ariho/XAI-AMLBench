# xai_explainer: audit-ready SAR narratives from GNN explanations

Turns a GNNExplainer subgraph (from `gnn_aml_core`) into a short, deterministic, audit-ready Suspicious Activity
Report narrative -- and guarantees that every number, date and account id it contains is traceable back to the
underlying evidence, so it can never accidentally invent a fact.

```python
from xai_explainer import build_facts, render_fact_sheet, template_narrative, validate_narrative

# explanation: the dict returned by gnn_aml_core's /explain endpoint (account_id, risk_score,
# reporting_threshold, nodes, edges, top_features) -- see notebooks/quickstart.ipynb for a full example.
facts = build_facts(explanation)
narrative = template_narrative(facts)
ok, problems = validate_narrative(narrative, render_fact_sheet(facts))
assert ok, problems   # the validator is the safety net: it fails the narrative rather than trust it blindly
```

## How it stays evidence-only

1. **`infer_typology`** classifies the pattern using transparent structural rules over the explained subgraph
   (fan-in hubs, cycles, shell-account chains, cross-border bidirectional bursts) -- not a second model, so the
   classification itself is auditable.
2. **`build_facts`** reduces the explanation to a small dict of pre-formatted, verified facts: dates, dollar
   amounts, account counts, country lists. Nothing here is generated language yet.
3. **`template_narrative`** (default, deterministic, no LLM) or an LLM backend (`build_messages`, if configured)
   turns the fact sheet into 3-4 sentences of plain-language narrative.
4. **`validate_narrative`** re-parses the generated text and rejects it if it contains any number, date or
   account id that is not present in the fact sheet, any of a short list of conclusive/accusatory words
   ("guilty", "convicted", ...), the wrong sentence count, or list/paragraph formatting. If the LLM backend's
   output fails validation, the caller falls back to `template_narrative`, which always passes.

This means the LLM (when used) can only ever paraphrase the fact sheet -- it cannot introduce a fact that was not
already verified.

## Typologies recognised

`smurfing` (many senders fan-in to one hub, or amounts clustered just under the reporting threshold),
`scatter_gather` (one-directional fan-out through a shared set of intermediaries), `cyclic_loop` (a directed
cycle among the accounts in the subgraph), `shell_company` (2+ shell-type accounts in the chain),
`cross_border_velocity` (10+ transactions directly touching one subject, genuinely bidirectional, spanning 3+
countries within about 5 days), or `unclassified` if none of these structural signatures match.

## Audit trail

Every generated narrative is hashed (`narrative_hash`) so a stored hash can later prove a narrative has not been
altered. `audit_log.py` (an append-only, tamper-evident log of every alert, model decision and human review) is
a planned addition -- see the project's brief scorecard (`check_brief_v1.py`, section E) for its current status.

## A note on accuracy

`infer_typology`'s structural rules are checked against every pattern the generator injects, across many random
seeds (see `tests/test_sar_generator.py`); as of this version it recovers the correct typology in more than 99.9%
of cases. This heuristic and the generator in `aml_synth` must be kept in sync: a change to how a typology's
patterns are shaped (their size, timing, or amount distribution) can silently break the corresponding structural
check here without either module raising an error. If you change `aml_synth/graph_generator.py`, re-run
`pytest tests/test_sar_generator.py -v` before assuming narratives are still classifying correctly.
