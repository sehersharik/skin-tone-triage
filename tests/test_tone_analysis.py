"""Phase 4 checks: standardisation, the regression's tone effect, seed configs, photo factors."""
import numpy as np
import pandas as pd
import pytest

from src.tone_analysis import (design_matrix, fit_coefs, load_predictions,
                               standardised_accuracy)
from src.utils import load_config

SEEDS_CONFIG = load_config("configs/seeds.yaml")


def two_groups_different_mix():
    """Group A is mostly the easy class, group B mostly the hard one. Within each class
    both groups are equally accurate, so any crude gap is pure condition mix."""
    y = np.array(["easy"] * 80 + ["hard"] * 20 + ["easy"] * 20 + ["hard"] * 80)
    group = np.array(["A"] * 100 + ["B"] * 100)
    # recall: easy 0.9, hard 0.3 in both groups
    correct = np.concatenate([np.arange(80) < 72, np.arange(20) < 6,
                              np.arange(20) < 18, np.arange(80) < 24]).astype(float)
    return y, group, correct


def test_standardised_accuracy_removes_a_pure_mix_gap():
    y, group, correct = two_groups_different_mix()
    crude_a, crude_b = correct[group == "A"].mean(), correct[group == "B"].mean()
    assert np.isclose(crude_a - crude_b, 0.78 - 0.42)          # big crude gap from mix only
    weights = {"easy": 0.5, "hard": 0.5}
    std_a = standardised_accuracy(correct, y, group == "A", weights)
    std_b = standardised_accuracy(correct, y, group == "B", weights)
    assert np.isclose(std_a, 0.6) and np.isclose(std_b, 0.6)    # 0.5 x 0.9 + 0.5 x 0.3


def test_standardised_equals_crude_when_mix_matches():
    y, group, correct = two_groups_different_mix()
    a = group == "A"
    weights = {"easy": 0.8, "hard": 0.2}                        # group A's own mix
    assert np.isclose(standardised_accuracy(correct, y, a, weights), correct[a].mean())


def test_regression_separates_tone_from_condition():
    rng = np.random.default_rng(0)
    y, group, _ = two_groups_different_mix()
    y, group = np.tile(y, 10), np.tile(group, 10)
    # correctness depends on the condition only, never on the group
    correct = (rng.random((3, len(y))) < np.where(y == "easy", 0.9, 0.3)).astype(float)
    df = pd.DataFrame({"tone": group, "label": y})
    X = design_matrix(df, "tone", ["A", "B"], ["easy", "hard"], "easy")
    tone_only = np.exp(fit_coefs(X[["tone: B"]].to_numpy(), correct)[0])
    adjusted = np.exp(fit_coefs(X.to_numpy(), correct)[0])
    assert tone_only < 0.4             # unadjusted: group B looks much worse
    assert 0.8 < adjusted < 1.25       # adjusted for condition: no tone effect


def test_seeds_config_reruns_the_phase3_baseline():
    base = load_config("configs/baseline.yaml")
    assert len(SEEDS_CONFIG["seeds"]) >= 3
    assert SEEDS_CONFIG["train"]["epochs"] == 5
    for key in ("images", "model"):
        assert SEEDS_CONFIG[key] == base[key]
    assert {k: v for k, v in SEEDS_CONFIG["train"].items() if k != "epochs"} == \
        {k: v for k, v in base["train"].items() if k not in ("epochs", "candidate_epochs",
                                                             "budget_fold")}


def test_seed_config_fills_paths():
    pytest.importorskip("torch")
    from src.train import seed_config
    cfg = seed_config(SEEDS_CONFIG, 43)
    assert cfg["seed"] == 43 and "seeds" not in cfg
    assert cfg["paths"]["predictions_csv"].endswith("seed43_oof.csv")
    assert cfg["paths"]["fold_outputs_dir"].endswith("seed43")
    assert "{seed}" not in str(cfg["paths"])
    assert SEEDS_CONFIG["seed"] == 42  # the shared config is not modified


def test_analysis_refuses_predictions_with_test_cases(tmp_path):
    cases = pd.DataFrame({"case_id": ["01", "02", "03"], "label": ["a", "b", "a"],
                          "split": ["cv", "cv", "test"]})
    cases.to_csv(tmp_path / "cases.csv", index=False)
    oof = pd.DataFrame({"case_id": ["01", "02", "03"], "prob_a": [0.9, 0.2, 0.5],
                        "prob_b": [0.1, 0.8, 0.5], "label": ["a", "b", "a"]})
    oof.to_csv(tmp_path / "seed1.csv", index=False)
    config = {"seeds": [1], "paths": {"cases_csv": tmp_path / "cases.csv",
                                      "predictions_csv": str(tmp_path / "seed{seed}.csv")}}
    with pytest.raises(AssertionError, match="CV pool"):
        load_predictions(config)
    oof[oof.case_id != "03"].to_csv(tmp_path / "seed1.csv", index=False)  # drop the test case
    cv, classes, probs = load_predictions(config)
    assert list(cv.case_id) == ["01", "02"] and classes == ["a", "b"] and probs.shape == (1, 2, 2)


def test_grey_stats():
    pytest.importorskip("cv2")
    from src.image_factors import grey_stats
    flat = np.full((64, 64), 100, np.uint8)
    assert grey_stats(flat) == {"brightness": 100.0, "contrast": 0.0, "sharpness": 0.0}
    checker = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
    blurred = np.full((64, 64), 128, np.uint8)
    blurred[:, 32:] = 160
    assert grey_stats(checker)["sharpness"] > grey_stats(blurred)["sharpness"]
