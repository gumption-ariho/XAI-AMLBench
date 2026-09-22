"""xai_explainer.extractor: turns a trained GNN plus its stored graph into a JSON-ready "explanation" -- the
dict shape xai_explainer.sar_generator.build_facts expects (see notebooks/quickstart.ipynb).

This is the brief's Module 3.1 ("Subgraph Extraction"), pulled out of the FastAPI service (gnn_aml_core.main)
so it can be tested, reused and audited on its own. PyTorch and torch_geometric are imported lazily inside
extract_explanation(), so `import xai_explainer.extractor` itself never requires them -- only calling the
function does, matching the rest of xai_explainer.
"""
from __future__ import annotations

from typing import Any


def extract_explanation(model: Any, graph: dict, feature_names: list[str], account_id: str, *, hops: int,
                        reporting_threshold: float, top_k: int = 40, epochs: int = 100,
                        max_explain_edges: int = 8000) -> dict:
    """Run GNNExplainer on `account_id`'s neighbourhood and return the input `xai_explainer.build_facts` expects.

    model: a trained GATv2Detector (only GATv2 is wired to GNNExplainer here; RGCN is not supported by this path).
    graph: the dict produced by `gnn_aml_core.train.arrays_from_frames` (also what `models/graph.pt` stores):
        x, edge_index, edge_attr, edge_type            -- the bidirectional graph tensors/arrays
        account_ids, account_type, country              -- per-node metadata, in the same order as x
        tx_ids, tx_amount, tx_timestamp, tx_cross_border -- per-transaction metadata (length n_tx, NOT 2*n_tx:
                                                             indices below refer to the ORIGINAL transaction)
        n_tx                                             -- int, the number of original (non-reversed) transactions
    feature_names: names for each column of `x`, in order (from the model checkpoint's "feature_names").
    hops: how many hops of neighbourhood to pull (normally the model's own number of message-passing layers).
    max_explain_edges: if the h-hop neighbourhood has more directed edges than this (a very large hub account),
        fall back to a 1-hop neighbourhood so GNNExplainer stays fast.

    Returns {"account_id", "risk_score", "reporting_threshold", "nodes", "edges", "top_features"}.
    Raises KeyError if account_id is not in graph["account_ids"].
    """
    import torch
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph

    from gnn_aml_core.features import merge_reverse_copies

    def as_tensor(v):
        return v if torch.is_tensor(v) else torch.as_tensor(v)

    ids = list(graph["account_ids"])
    if account_id not in ids:
        raise KeyError(f"unknown account_id: {account_id!r}")
    idx = ids.index(account_id)

    x = as_tensor(graph["x"])
    edge_index = as_tensor(graph["edge_index"])
    edge_attr = as_tensor(graph["edge_attr"])
    edge_type = as_tensor(graph["edge_type"])
    n = x.size(0)

    # The stored graph is bidirectional (every transaction also has a reversed copy), so one k-hop query already
    # returns the complete neighbourhood; a very large hub falls back to 1 hop so GNNExplainer stays fast.
    subset, ei, mapping, emask = k_hop_subgraph(idx, hops, edge_index, relabel_nodes=True, num_nodes=n)
    if ei.size(1) > max_explain_edges:
        subset, ei, mapping, emask = k_hop_subgraph(idx, 1, edge_index, relabel_nodes=True, num_nodes=n)
    ea, et = edge_attr[emask], edge_type[emask]
    x_sub = x[subset]
    center = int(mapping[0])

    with torch.no_grad():
        score = float(torch.sigmoid(model(x_sub, ei, ea, et)[center]))

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )
    expl = explainer(x_sub, ei, index=center, edge_attr=ea, edge_type=et)
    edge_imp = expl.edge_mask.detach().cpu()
    feat_imp = expl.node_mask[center].abs().detach().cpu()

    # emask marks which of the 2*n_tx bidirectional edges fall in this neighbourhood; merge_reverse_copies folds
    # each transaction's forward and reversed copy back into one edge with its true (original) direction.
    n_tx = int(graph["n_tx"])
    global_edge_ids = emask.nonzero().view(-1).cpu().numpy()
    folded = merge_reverse_copies(global_edge_ids, ei[0].cpu().numpy(), ei[1].cpu().numpy(), edge_imp.numpy(), n_tx, top_k)

    node_ids = {center}
    edges = []
    for tx, s_local, d_local, imp in folded:
        node_ids.update((s_local, d_local))
        edges.append({
            "tx_id": graph["tx_ids"][tx],
            "src": ids[int(subset[s_local])],
            "dst": ids[int(subset[d_local])],
            "amount": round(float(graph["tx_amount"][tx]), 2),
            "timestamp": int(graph["tx_timestamp"][tx]),
            "cross_border": int(graph["tx_cross_border"][tx]),
            "importance": round(imp, 5),
        })

    nodes = []
    for local_idx in sorted(node_ids):
        global_idx = int(subset[local_idx])
        nodes.append({"account_id": ids[global_idx], "account_type": graph["account_type"][global_idx],
                      "country": graph["country"][global_idx]})

    total = float(feat_imp.sum()) or 1.0
    top = torch.argsort(feat_imp, descending=True)[:6].tolist()
    top_features = [{"feature": feature_names[i], "weight": round(float(feat_imp[i]) / total, 4)}
                    for i in top if feat_imp[i] > 0]

    return {
        "account_id": account_id, "risk_score": round(score, 5), "reporting_threshold": reporting_threshold,
        "nodes": nodes, "edges": edges, "top_features": top_features,
    }
