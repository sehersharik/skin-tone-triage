"""Phase 7: score the frozen method on the test set, once.

Everything about the method was fixed on cross-validation results before this
script was written: the model file is models/final_baseline.pt (the baseline
trained on all CV cases). This script only predicts the test cases, averages
photo probabilities within each case, and computes the same case-level metrics
as the CV evaluation, overall and per FST and Monk group, with 95% bootstrap
CIs (cases resampled). It also compares the test numbers with the seed-42 CV
out-of-fold numbers.

Guard against reusing the test set: if reports/metrics.json already holds a
`test_final` result, or the test predictions file exists, the script stops.
`--overwrite-test-result` lets it run again; use it only when the owner says so.
A rerun with identical inputs gives identical numbers (CPU, fixed bootstrap seed).

Usage:
    python -m src.test_eval --config configs/test.yaml
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.data import case_images, split_fingerprint
from src.evaluate import (EFST_ORDER, METRICS, MONK_ORDER, ci, evaluate_groups, evaluate_subset,
                          plot_confusion, plot_groups)
from src.models import build_model
from src.train import get_device, predict_cases
from src.utils import load_config, save_metrics, set_seed, start_run

SECTION = "test_final"
PROVENANCE = ("evaluated_at", "git_commit", "run_dir")   # the only fields allowed to differ on a rerun


def load_test_set(paths):
    """Read cases.csv, check it matches the committed splits, return the test cases only."""
    cases = pd.read_csv(paths["cases_csv"], dtype={"case_id": str})
    expected = Path(paths["split_fingerprint"]).read_text().strip()
    if split_fingerprint(cases) != expected:
        raise SystemExit("cases.csv does not match reports/split_fingerprint.txt: "
                         "the splits differ from the committed ones")
    return cases[cases.split == "test"].reset_index(drop=True)


def check_not_done(paths, overwrite):
    """Refuse to reuse the test set: stop if a test result already exists, unless told to."""
    metrics_path = Path(paths["metrics_json"])
    done = []
    if metrics_path.exists() and SECTION in json.loads(metrics_path.read_text()):
        done.append(f"{metrics_path} already has a '{SECTION}' result")
    if Path(paths["test_predictions_csv"]).exists():
        done.append(f"{paths['test_predictions_csv']} already exists")
    if done and not overwrite:
        raise SystemExit("The test set has already been scored (" + "; ".join(done) + "). "
                         "It is scored once, for the frozen method. Rerun with "
                         "--overwrite-test-result only on the owner's explicit instruction.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_test(config, cases, classes, device):
    """Case-level probabilities from the final model (mean over each case's photos)."""
    paths = config["paths"]
    model = build_model(config["model"]["name"], len(classes), pretrained=False)
    model.load_state_dict(torch.load(paths["final_model"], map_location="cpu"))
    model.to(device)
    cases = cases.assign(label_idx=cases.label.map(classes.index))
    images = case_images(cases).merge(cases[["case_id", "label_idx"]], on="case_id")
    preds = predict_cases(model, images, config, classes, device)
    preds = preds.merge(cases[["case_id", "label"]], on="case_id", validate="one_to_one")
    preds["pred"] = np.asarray(classes)[preds[[f"prob_{c}" for c in classes]].to_numpy()
                                        .argmax(axis=1)]
    return preds, len(images)


def small_groups(df, group_col, order, classes, min_cases):
    out = {}
    for g in order:
        sub = df[df[group_col] == g]
        absent = [c for c in classes if not (sub.label == c).any()]
        out[g] = {"n_cases": int(len(sub)), "classes_absent": absent,
                  "weak_evidence": bool(len(sub) < min_cases or absent)}
    return out


def score(preds, cases, cv_preds, classes, ev, seed, min_cases):
    """All test numbers, from the case predictions alone. Same inputs -> same output."""
    rng = np.random.default_rng(seed)
    df = preds.merge(cases[["case_id", "efst_group", "monk_us_group"]], on="case_id",
                     validate="one_to_one").sort_values("case_id", ignore_index=True)
    overall, test_boot = evaluate_subset(df, classes, ev, rng)
    cm = confusion_matrix(df.label, df.pred, labels=classes)
    by_efst, efst_gaps = evaluate_groups(df, "efst_group", EFST_ORDER, classes, ev, rng)
    by_monk, monk_gaps = evaluate_groups(df, "monk_us_group", MONK_ORDER, classes, ev, rng)

    # Test vs CV: two independent sets of cases, so their bootstrap samples are independent
    cv_preds = cv_preds.sort_values("case_id", ignore_index=True)
    cv_overall, cv_boot = evaluate_subset(cv_preds, classes, ev, rng)
    comparison = {}
    for m in METRICS:
        low, high = ci(test_boot[m].to_numpy() - cv_boot[m].to_numpy(), ev["ci_level"])
        comparison[m] = {"test": overall[m], "cv": cv_overall[m],
                         "test_minus_cv": round(overall[m]["value"] - cv_overall[m]["value"], 4),
                         "ci_low": low, "ci_high": high, "outside_noise": bool(low > 0 or high < 0)}
    return {
        "overall": overall,
        "per_class_recall": {c: (round(float(cm[i, i] / cm[i].sum()), 4) if cm[i].sum() else None)
                             for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": cm.tolist()},
        "by_efst_group": by_efst,
        "efst_gaps": efst_gaps,
        "by_monk_group": by_monk,
        "monk_gaps": monk_gaps,
        "small_groups": {"efst": small_groups(df, "efst_group", EFST_ORDER, classes, min_cases),
                         "monk": small_groups(df, "monk_us_group", MONK_ORDER, classes, min_cases)},
        "cv_comparison": {"cv_source": "seed-42 out-of-fold predictions on the CV pool "
                                       "(the same procedure, models trained on 4 of 5 folds)",
                          "metrics": comparison},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-test-result", action="store_true",
                        help="score the test set again although a result exists (owner's call)")
    args = parser.parse_args()
    config = load_config(args.config)
    paths, ev = config["paths"], config["evaluation"]
    check_not_done(paths, args.overwrite_test_result)
    set_seed(config["seed"])
    torch.backends.cudnn.deterministic = True

    info = json.loads(Path(paths["final_model_info"]).read_text())
    classes = info["classes"]
    cases = load_test_set(paths)
    device = get_device(config["train"]["device"])
    run_dir = start_run(paths["runs_dir"], f"{config['experiment']}", config)
    print(f"Scoring {len(cases)} test cases on {device} with {paths['final_model']}")
    preds, n_images = predict_test(config, cases, classes, device)

    cv_preds = pd.read_csv(paths["cv_predictions_csv"], dtype={"case_id": str})
    numbers = score(preds, cases, cv_preds, classes, ev, config["seed"],
                    config["test"]["small_group_cases"])
    out = Path(paths["test_predictions_csv"])    # written only once scoring has succeeded
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out, index=False)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or None
    section = {
        "description": "Phase 7. The one-time test-set evaluation of the frozen method "
                       "(baseline EfficientNet-B0, ImageNet weights, no augmentation, 5 epochs, "
                       "seed 42) trained on all CV cases. Case level; 95% bootstrap CIs resample "
                       "cases.",
        "config": args.config,
        "model_file": paths["final_model"],
        "model_sha256": sha256(paths["final_model"]),
        "model_info": {k: info[k] for k in ("seed", "epochs", "n_cases", "n_images",
                                            "torch_version")},
        "n_cases": int(len(cases)), "n_images": int(n_images),
        "device": str(device), "torch_version": torch.__version__,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit, "run_dir": str(run_dir),
        **numbers,
    }
    metrics_path = Path(paths["metrics_json"])
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[SECTION] = section
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    save_metrics(run_dir, section)

    figures = Path(paths["figures_dir"])
    plot_confusion(numbers["confusion_matrix"]["rows_true_cols_pred"], classes,
                   figures / "test_confusion_matrix.png",
                   title="Frozen baseline on the TEST set (cases)\ncount and % of true class")
    plot_groups(numbers, figures / "test_by_skin_tone.png",
                title="Frozen baseline on the TEST set by skin-tone group "
                      "(95% bootstrap CI; dashed = overall)")
    print(json.dumps({"overall": numbers["overall"], "cv_comparison": numbers["cv_comparison"]},
                     indent=2))
    print(f"Wrote {metrics_path} ({SECTION}), {out} and the test figures")


if __name__ == "__main__":
    main()
