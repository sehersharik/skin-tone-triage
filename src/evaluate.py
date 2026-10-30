"""Case-level evaluation of out-of-fold (CV) predictions, overall and per skin-tone group.

Metrics (all at the case level, one prediction per case):
- balanced accuracy: mean recall over the classes present in the evaluated cases
- macro F1: unweighted mean F1 over the classes present
- macro one-vs-rest AUC: mean AUC over classes that have both positives and negatives

95% confidence intervals come from bootstrapping cases (resampling cases with
replacement). A gap between two skin-tone groups is "outside the noise" only if
the bootstrap CI of the difference excludes 0.

Every number is written to reports/metrics.json; figures are drawn from it.

Usage:
    python -m src.evaluate --config configs/baseline.yaml
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from src.utils import load_config

METRICS = ["balanced_accuracy", "macro_f1", "macro_auc_ovr"]
EFST_ORDER = ["FST I-II", "FST III-IV", "FST V-VI"]
MONK_ORDER = ["Monk 1-3", "Monk 4-6", "Monk 7-10"]


def macro_auc_ovr(y_true, probs, classes):
    aucs = [roc_auc_score(y_true == c, probs[:, i]) for i, c in enumerate(classes)
            if 0 < (y_true == c).sum() < len(y_true)]
    return float(np.mean(aucs)) if aucs else np.nan


def case_metrics(y_true, probs, classes):
    """Metrics for one set of case predictions. `probs` columns follow `classes`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(classes)[probs.argmax(axis=1)]
    present = [c for c in classes if (y_true == c).any()]
    recalls = [(y_pred[y_true == c] == c).mean() for c in present]
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro",
                                   zero_division=0)),
        "macro_auc_ovr": macro_auc_ovr(y_true, probs, classes),
    }


def bootstrap_metrics(y_true, probs, classes, n_bootstrap, rng):
    """Recompute the metrics on `n_bootstrap` resamples of the cases."""
    y_true = np.asarray(y_true)
    idx = rng.integers(0, len(y_true), size=(n_bootstrap, len(y_true)))
    return pd.DataFrame([case_metrics(y_true[i], probs[i], classes) for i in idx])


def ci(samples, ci_level):
    tail = (1 - ci_level) / 2 * 100
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return round(float(low), 4), round(float(high), 4)


def summarise(point, samples, ci_level):
    low, high = ci(samples, ci_level)
    return {"value": round(point, 4), "ci_low": low, "ci_high": high}


def evaluate_subset(df, classes, eval_cfg, rng):
    """Point metrics + bootstrap CIs for a set of cases; also returns the bootstrap samples."""
    probs = df[[f"prob_{c}" for c in classes]].to_numpy()
    point = case_metrics(df.label, probs, classes)
    boot = bootstrap_metrics(df.label, probs, classes, eval_cfg["n_bootstrap"], rng)
    result = {"n_cases": len(df),
              "cases_per_class": df.label.value_counts().reindex(classes, fill_value=0).to_dict()}
    result.update({m: summarise(point[m], boot[m], eval_cfg["ci_level"]) for m in METRICS})
    return result, boot


def evaluate_groups(df, group_col, order, classes, eval_cfg, rng):
    """Per-group metrics and pairwise gaps. Cases with no group label are excluded,
    and a group with no cases is reported with n_cases = 0 and no metrics."""
    results, boots = {}, {}
    for group in order:
        subset = df[df[group_col] == group]
        if subset.empty:
            results[group] = {"n_cases": 0}
            continue
        results[group], boots[group] = evaluate_subset(subset, classes, eval_cfg, rng)
    gaps = {}
    for a, b in combinations([g for g in order if g in boots], 2):
        gaps[f"{b} minus {a}"] = {}
        for m in METRICS:
            diff = results[b][m]["value"] - results[a][m]["value"]
            low, high = ci(boots[b][m].to_numpy() - boots[a][m].to_numpy(), eval_cfg["ci_level"])
            gaps[f"{b} minus {a}"][m] = {
                "difference": round(diff, 4), "ci_low": low, "ci_high": high,
                "outside_noise": bool(low > 0 or high < 0)}
    results["excluded_missing_group"] = int(df[group_col].isna().sum())
    return results, gaps


def plot_confusion(cm, classes, path):
    cm = np.asarray(cm)
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if row_pct[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="predicted", ylabel="true",
           title="Baseline confusion matrix (CV out-of-fold, cases)\ncount and % of true class")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_groups(section, path):
    """Point estimate and 95% CI per skin-tone group, one panel per metric."""
    rows = [(g, "eFST", section["by_efst_group"][g]) for g in EFST_ORDER] + \
           [(g, "Monk (US)", section["by_monk_group"][g]) for g in MONK_ORDER]
    rows = [r for r in rows if r[2]["n_cases"] > 0]
    colors = {"eFST": "#2a78d6", "Monk (US)": "#eb6834"}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 3.8), sharey=True)
    for ax, m in zip(axes, METRICS):
        for y, (group, kind, res) in enumerate(rows):
            v = res[m]
            ax.plot([v["ci_low"], v["ci_high"]], [y, y], color=colors[kind], lw=2)
            ax.plot(v["value"], y, "o", color=colors[kind], ms=7,
                    label=kind if group in (EFST_ORDER[0], MONK_ORDER[0]) else None)
        overall = section["overall"][m]["value"]
        ax.axvline(overall, color="#52514e", lw=1, ls="--")
        ax.set(title=m.replace("_", " "), xlim=(0, 1.03))
    axes[0].set_yticks(range(len(rows)),
                       [f"{g} (n={r['n_cases']})" for g, _, r in rows])
    axes[0].invert_yaxis()
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Baseline by skin-tone group (CV out-of-fold, 95% bootstrap CI; dashed = overall)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths, eval_cfg = config["paths"], config["evaluation"]
    rng = np.random.default_rng(config["seed"])

    cases = pd.read_csv(paths["cases_csv"], dtype={"case_id": str})
    oof = pd.read_csv(paths["predictions_csv"], dtype={"case_id": str})
    train_log = json.loads(Path(paths["train_log_json"]).read_text())
    classes = train_log["classes"]

    # Safety: only CV cases, each with exactly one out-of-fold prediction
    cv_cases = cases[cases.split == "cv"]
    assert oof.case_id.is_unique and set(oof.case_id) == set(cv_cases.case_id), \
        "OOF predictions must cover exactly the CV pool"
    oof = oof.merge(cases[["case_id", "efst_group", "monk_us_group"]], on="case_id",
                    how="left", validate="one_to_one")

    overall, _ = evaluate_subset(oof, classes, eval_cfg, rng)
    cm = confusion_matrix(oof.label, oof.pred, labels=classes)
    by_efst, efst_gaps = evaluate_groups(oof, "efst_group", EFST_ORDER, classes, eval_cfg, rng)
    by_monk, monk_gaps = evaluate_groups(oof, "monk_us_group", MONK_ORDER, classes, eval_cfg, rng)

    section = {
        "description": "Baseline EfficientNet-B0, no augmentation. Case-level out-of-fold "
                       "predictions on the CV pool; the test set was not used.",
        "config": args.config,
        "classes": classes,
        "epoch_budget": train_log["epoch_budget"],
        "timing": train_log["timing"],
        "overall": overall,
        "per_class_recall": {c: round(float(cm[i, i] / cm[i].sum()), 4)
                             for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes, "rows_true_cols_pred": cm.tolist()},
        "by_efst_group": by_efst,
        "efst_gaps": efst_gaps,
        "by_monk_group": by_monk,
        "monk_gaps": monk_gaps,
    }

    metrics_path = Path(paths["metrics_json"])
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["baseline_cv"] = section
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    figures = Path(paths["figures_dir"])
    plot_confusion(cm, classes, figures / "baseline_confusion_matrix.png")
    plot_groups(section, figures / "baseline_by_skin_tone.png")
    print(json.dumps({k: section[k] for k in ("overall", "efst_gaps")}, indent=2))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
