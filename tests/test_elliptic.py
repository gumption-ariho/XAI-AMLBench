"""Tests for gnn_aml_core.elliptic: the loader and adapter for the real-world Elliptic Bitcoin dataset. No
PyTorch needed (load_elliptic/elliptic_splits are pure numpy/pandas/scikit-learn), so these always run. A small,
synthetic stand-in dataset is generated with the EXACT real file format (headerless features file, a classes
file using the literal string "unknown" for unlabelled rows, a dangling edge referencing a non-existent
transaction id) -- this is not the real ~30MB Elliptic download, which this test suite never needs.
"""
import numpy as np
import pandas as pd
import pytest

from gnn_aml_core.elliptic import DEFAULT_TEMPORAL_CUTOFF, elliptic_splits, load_elliptic


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory):
    """A tiny Elliptic-shaped dataset: 60 transactions across 6 time steps, a realistic mix of illicit/licit/
    unknown labels, and one dangling edge (referencing a transaction id absent from the features file)."""
    d = tmp_path_factory.mktemp("elliptic_stub")
    rng = np.random.default_rng(0)
    n = 60
    tx_ids = np.arange(1000, 1000 + n)
    time_step = rng.integers(1, 7, n)  # 1..6
    features = rng.normal(size=(n, 10))

    with open(d / "elliptic_txs_features.csv", "w") as f:
        for i in range(n):
            row = [str(tx_ids[i]), str(time_step[i])] + [f"{v:.6f}" for v in features[i]]
            f.write(",".join(row) + "\n")

    classes = rng.choice(["1", "2", "unknown"], size=n, p=[0.15, 0.35, 0.50])
    pd.DataFrame({"txId": tx_ids, "class": classes}).to_csv(d / "elliptic_txs_classes.csv", index=False)

    n_edges = 90
    src = rng.choice(tx_ids, n_edges)
    dst = rng.choice(tx_ids, n_edges)
    edges = pd.DataFrame({"txId1": src, "txId2": dst})
    edges = pd.concat([edges, pd.DataFrame({"txId1": [999999], "txId2": [tx_ids[0]]})], ignore_index=True)
    edges.to_csv(d / "elliptic_txs_edgelist.csv", index=False)

    return d, tx_ids, classes, n_edges


class TestLoadElliptic:
    def test_missing_files_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="elliptic_txs_features.csv"):
            load_elliptic(tmp_path)

    def test_feature_matrix_shape(self, stub_dir):
        d, tx_ids, _, _ = stub_dir
        arrays = load_elliptic(d)
        assert arrays["x"].shape == (len(tx_ids), 10)

    def test_account_ids_are_clean_integers_not_floats(self, stub_dir):
        d, _, _, _ = stub_dir
        arrays = load_elliptic(d)
        assert all("." not in tid for tid in arrays["account_ids"])

    def test_class_remapping_illicit_is_one_licit_is_zero(self, stub_dir):
        # The single most important correctness property: a flipped label here would silently invert every
        # result this module ever produces. Cross-checked directly against the raw classes file.
        d, _, classes, _ = stub_dir
        raw = pd.read_csv(d / "elliptic_txs_classes.csv")
        arrays = load_elliptic(d)
        raw_illicit_ids = set(raw.loc[raw["class"].astype(str) == "1", "txId"])
        raw_licit_ids = set(raw.loc[raw["class"].astype(str) == "2", "txId"])
        for i, tid in enumerate(arrays["account_ids"]):
            if int(tid) in raw_illicit_ids:
                assert arrays["labeled_mask"][i] and arrays["y"][i] == 1.0
            if int(tid) in raw_licit_ids:
                assert arrays["labeled_mask"][i] and arrays["y"][i] == 0.0

    def test_unknown_rows_excluded_from_labeled_mask(self, stub_dir):
        d, _, classes, _ = stub_dir
        arrays = load_elliptic(d)
        assert int((~arrays["labeled_mask"]).sum()) == int((classes == "unknown").sum())

    def test_dangling_edge_is_dropped_not_crashed(self, stub_dir):
        d, _, _, n_edges = stub_dir
        arrays = load_elliptic(d)
        assert arrays["n_tx"] == n_edges  # the one dangling edge referencing txId 999999 is dropped

    def test_edge_index_is_bidirectional(self, stub_dir):
        d, _, _, n_edges = stub_dir
        arrays = load_elliptic(d)
        assert arrays["edge_index"].shape == (2, 2 * n_edges)

    def test_edge_attr_has_time_gap_and_direction_flag(self, stub_dir):
        d, _, _, n_edges = stub_dir
        arrays = load_elliptic(d)
        assert arrays["edge_attr"].shape == (2 * n_edges, 2)
        assert set(np.unique(arrays["edge_attr"][:, 1])) <= {0.0, 1.0}

    def test_edge_time_gap_matches_endpoint_time_steps(self, stub_dir):
        d, _, _, _ = stub_dir
        arrays = load_elliptic(d)
        n_tx = arrays["n_tx"]
        src, dst = arrays["src_tx"], arrays["dst_tx"]
        expected = np.abs(arrays["time_step"][src] - arrays["time_step"][dst])
        np.testing.assert_allclose(arrays["edge_attr"][:n_tx, 0], expected)

    def test_feature_names_match_column_count(self, stub_dir):
        d, _, _, _ = stub_dir
        arrays = load_elliptic(d)
        assert len(arrays["feature_names"]) == arrays["x"].shape[1]


class TestEllipticSplits:
    def test_random_split_uses_only_labelled_nodes_no_overlap(self, stub_dir):
        d, _, _, _ = stub_dir
        arrays = load_elliptic(d)
        tr, va, te = elliptic_splits(arrays, split="random", seed=1)
        all_idx = np.concatenate([tr, va, te])
        assert len(set(all_idx.tolist())) == len(all_idx)
        assert arrays["labeled_mask"][all_idx].all()

    def test_temporal_split_respects_the_cutoff(self, stub_dir):
        d, _, _, _ = stub_dir
        arrays = load_elliptic(d)
        tr, va, te = elliptic_splits(arrays, split="temporal", cutoff=3, seed=1)
        assert (arrays["time_step"][tr] <= 3).all()
        assert (arrays["time_step"][va] > 3).all()
        assert (arrays["time_step"][te] > 3).all()

    def test_temporal_split_uses_only_labelled_nodes(self, stub_dir):
        d, _, _, _ = stub_dir
        arrays = load_elliptic(d)
        tr, va, te = elliptic_splits(arrays, split="temporal", cutoff=3, seed=1)
        for idx in (tr, va, te):
            assert arrays["labeled_mask"][idx].all()

    def test_default_cutoff_matches_the_published_paper(self):
        assert DEFAULT_TEMPORAL_CUTOFF == 34

    def test_unknown_split_name_raises(self, stub_dir):
        d, _, _, _ = stub_dir
        arrays = load_elliptic(d)
        with pytest.raises(ValueError, match="split"):
            elliptic_splits(arrays, split="not_a_real_split")

    def test_same_seed_is_reproducible(self, stub_dir):
        d, _, _, _ = stub_dir
        arrays = load_elliptic(d)
        a = elliptic_splits(arrays, split="random", seed=7)
        b = elliptic_splits(arrays, split="random", seed=7)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
