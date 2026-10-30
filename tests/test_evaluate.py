import numpy as np
import pandas as pd

from src.evaluate import (bootstrap_metrics, case_metrics, evaluate_groups,
                          macro_auc_ovr)

CLASSES = ["a", "b", "c"]


def one_hot(labels, confidence=0.9):
    probs = np.full((len(labels), len(CLASSES)), (1 - confidence) / (len(CLASSES) - 1))
    for i, label in enumerate(labels):
        probs[i, CLASSES.index(label)] = confidence
    return probs


def test_perfect_predictions():
    y = np.array(["a", "b", "c", "a"])
    m = case_metrics(y, one_hot(y), CLASSES)
    assert m == {"balanced_accuracy": 1.0, "macro_f1": 1.0, "macro_auc_ovr": 1.0}


def test_balanced_accuracy_is_mean_recall_over_present_classes():
    y_true = np.array(["a", "a", "a", "a", "b", "b"])
    y_pred = np.array(["a", "a", "a", "b", "b", "a"])  # recall a = 0.75, b = 0.5
    m = case_metrics(y_true, one_hot(y_pred), CLASSES)
    assert np.isclose(m["balanced_accuracy"], 0.625)  # class c absent -> not averaged


def test_auc_skips_classes_without_positives():
    y = np.array(["a", "b", "a", "b"])
    assert macro_auc_ovr(y, one_hot(y), CLASSES) == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.choice(CLASSES, 300)
    y_pred = np.where(rng.random(300) < 0.7, y_true, rng.choice(CLASSES, 300))
    probs = one_hot(y_pred)
    point = case_metrics(y_true, probs, CLASSES)["balanced_accuracy"]
    boot = bootstrap_metrics(y_true, probs, CLASSES, 500, rng)["balanced_accuracy"]
    low, high = np.percentile(boot, [2.5, 97.5])
    assert low < point < high and high - low < 0.2


def test_gap_detected_only_when_real():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.choice(CLASSES, n)
    group = np.repeat(["g1", "g2"], n // 2)
    # g1: always right; g2: right 40% of the time -> a real gap
    correct = np.where(group == "g1", True, rng.random(n) < 0.4)
    pred = np.where(correct, y, [CLASSES[(CLASSES.index(c) + 1) % 3] for c in y])
    df = pd.DataFrame(one_hot(pred), columns=[f"prob_{c}" for c in CLASSES])
    df["label"], df["group"] = y, group
    _, gaps = evaluate_groups(df, "group", ["g1", "g2"], CLASSES,
                              {"n_bootstrap": 300, "ci_level": 0.95}, rng)
    assert gaps["g2 minus g1"]["balanced_accuracy"]["outside_noise"]

    # identical behaviour in both groups -> CI of the gap should include 0
    df["group"] = rng.permutation(group)
    df["label"] = y
    df[[f"prob_{c}" for c in CLASSES]] = one_hot(np.where(rng.random(n) < 0.7, y, pred))
    _, gaps = evaluate_groups(df, "group", ["g1", "g2"], CLASSES,
                              {"n_bootstrap": 300, "ci_level": 0.95}, rng)
    g = gaps["g2 minus g1"]["balanced_accuracy"]
    assert g["ci_low"] < 0 < g["ci_high"]
