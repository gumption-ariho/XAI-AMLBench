"""Tests for aml_synth.adversarial. The pure config-adaptation logic runs without torch or a real training
run; run_adaptation_round's full end-to-end test is slower (trains two real LogisticRegression models on real
generated graphs)."""
from aml_synth.adversarial import FEATURE_TO_TYPOLOGIES, adapted_config


class TestAdaptedConfig:
    def test_camouflage_boosted_when_a_mapped_feature_is_relied_on(self):
        base = {"n_accounts": 5000, "camouflage": 1.5}
        out = adapted_config(base, ["pass_through_match_ratio"], boost=2.0)
        assert out["camouflage"] == 3.0

    def test_original_dict_is_never_mutated(self):
        base = {"n_accounts": 5000, "camouflage": 1.5}
        adapted_config(base, ["pass_through_match_ratio"], boost=2.0)
        assert base["camouflage"] == 1.5

    def test_unmapped_feature_leaves_camouflage_unchanged(self):
        base = {"n_accounts": 5000, "camouflage": 1.5}
        out = adapted_config(base, ["some_unmapped_feature_xyz"], boost=2.0)
        assert out["camouflage"] == 1.5

    def test_missing_camouflage_key_defaults_to_1point5_before_boosting(self):
        base = {"n_accounts": 5000}
        out = adapted_config(base, ["pass_through_match_ratio"], boost=2.0)
        assert out["camouflage"] == 3.0

    def test_non_camouflage_keys_pass_through_unchanged(self):
        base = {"n_accounts": 5000, "seed": 7, "camouflage": 1.5}
        out = adapted_config(base, ["pass_through_match_ratio"], boost=2.0)
        assert out["n_accounts"] == 5000
        assert out["seed"] == 7


class TestFeatureToTypologyMapping:
    def test_every_mapped_feature_name_is_a_real_feature(self):
        from gnn_aml_core.features import FEATURE_NAMES
        for feat in FEATURE_TO_TYPOLOGIES:
            assert feat in FEATURE_NAMES, f"{feat} is not a real feature name"

    def test_every_mapped_typology_name_is_real(self):
        from aml_synth.graph_generator import TYPOLOGIES
        for typologies in FEATURE_TO_TYPOLOGIES.values():
            for t in typologies:
                assert t in TYPOLOGIES, f"{t} is not a real typology name"


class TestRunAdaptationRoundEndToEnd:
    def test_returns_all_expected_keys(self):
        from aml_synth.adversarial import run_adaptation_round
        res = run_adaptation_round(n_accounts=800, seed=1, top_k=3)
        for key in ("gen1_auc", "gen2_auc_same_model", "gen2_auc_retrained", "relied_on_features",
                   "degradation", "recovery"):
            assert key in res

    def test_aucs_are_in_a_sane_range(self):
        from aml_synth.adversarial import run_adaptation_round
        res = run_adaptation_round(n_accounts=800, seed=1, top_k=3)
        for key in ("gen1_auc", "gen2_auc_same_model", "gen2_auc_retrained"):
            assert 0.4 <= res[key] <= 1.0

    def test_relied_on_features_has_the_requested_count(self):
        from aml_synth.adversarial import run_adaptation_round
        res = run_adaptation_round(n_accounts=800, seed=1, top_k=3)
        assert len(res["relied_on_features"]) == 3
