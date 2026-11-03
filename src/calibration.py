"""How well do the model's probabilities match how often it is right? CV only.

Uses the seed-42 out-of-fold predictions (each case predicted by a model that
never saw it). The test set is not used.

- Reliability table: cases binned by the top-class probability ("confidence");
  per bin, the mean confidence and the share actually correct. A calibrated
  model has accuracy = confidence in every bin.
- Expected calibration error (ECE): the gap |accuracy - confidence| averaged
  over bins, weighted by how many cases each bin holds. 0 = perfectly calibrated.
- Overconfidence gap: mean confidence minus accuracy (> 0 = overconfident).
- Brier score: mean over cases of sum_k (p_k - y_k)^2, where y is the one-hot
  true class. Lower is better; it rewards being right AND being sure.
- Temperature scaling, cross-fitted: probabilities are sharpened or softened as
  p_k^(1/T) (renormalised). T is fitted (lowest log-loss) on 4 CV folds and
  applied to the 5th, so every case is scaled by a T that never saw it.
- Mean predicted probability per class vs the real share of each class: the
  class-weighted loss trains the model as if all classes were equally common.

95% CIs come from resampling cases. Writes `calibration_cv` to metrics.json.

Usage:
    python -m src.calibration --config configs/app.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate import ci
from src.tone_analysis import style
from src.utils import load_config

RAW, SCALED = "#2a78d6", "#eb6834"


def reliability_table(confidence, correct, n_bins):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n_cases": int(mask.sum()),
                     "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                     "accuracy": float(correct[mask].mean()) if mask.any() else None})
    return rows


def ece(confidence, correct, n_bins):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def brier(probs, y_idx):
    onehot = np.eye(probs.shape[1])[y_idx]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def scale(probs, temperature):
    logp = np.log(np.clip(probs, 1e-12, 1)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(probs, y_idx, grid):
    """The T on `grid` with the lowest mean log-loss."""
    losses = [-np.log(np.clip(scale(probs, t)[np.arange(len(y_idx)), y_idx], 1e-12, 1)).mean()
              for t in grid]
    return float(grid[int(np.argmin(losses))])


def cross_fitted_scaling(probs, y_idx, folds, grid):
    scaled, temps = np.empty_like(probs), {}
    for f in np.unique(folds):
        t = fit_temperature(probs[folds != f], y_idx[folds != f], grid)
        scaled[folds == f] = scale(probs[folds == f], t)
        temps[int(f)] = round(t, 3)
    return scaled, temps


def summary(probs, y_idx, n_bins):
    conf, correct = probs.max(axis=1), (probs.argmax(axis=1) == y_idx).astype(float)
    return {"ece": ece(conf, correct, n_bins), "brier": brier(probs, y_idx),
            "mean_confidence": float(conf.mean()), "accuracy": float(correct.mean()),
            "overconfidence_gap": float(conf.mean() - correct.mean())}


def with_ci(probs_by_name, y_idx, n_bins, n_bootstrap, ci_level, rng):
    """Point values and CIs for each set of probabilities, on shared resamples (so paired)."""
    points = {n: summary(p, y_idx, n_bins) for n, p in probs_by_name.items()}
    idx = rng.integers(0, len(y_idx), size=(n_bootstrap, len(y_idx)))
    boots = {n: pd.DataFrame([summary(p[i], y_idx[i], n_bins) for i in idx])
             for n, p in probs_by_name.items()}
    out = {}
    for n in probs_by_name:
        out[n] = {k: {"value": round(v, 4), **dict(zip(("ci_low", "ci_high"),
                                                       ci(boots[n][k], ci_level)))}
                  for k, v in points[n].items()}
    names = list(probs_by_name)
    if len(names) == 2:
        a, b = names
        out[f"{b}_minus_{a}"] = {}
        for k in ("ece", "brier"):
            low, high = ci(boots[b][k] - boots[a][k], ci_level)
            out[f"{b}_minus_{a}"][k] = {"difference": round(points[b][k] - points[a][k], 4),
                                        "ci_low": low, "ci_high": high,
                                        "outside_noise": bool(low > 0 or high < 0)}
    return out


def plot(tables, path):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1, ls="--", label="perfect calibration")
    for (name, rows), color in zip(tables.items(), (RAW, SCALED)):
        rows = [r for r in rows if r["n_cases"] > 0]
        ax.plot([r["mean_confidence"] for r in rows], [r["accuracy"] for r in rows], "-o",
                color=color, lw=2, ms=8, label=name)
    for r in tables["raw"]:
        if r["n_cases"]:
            ax.annotate(f"n={r['n_cases']}", (r["mean_confidence"], r["accuracy"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=7, color="#52514e")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean top-class probability in the bin",
           ylabel="share of cases predicted correctly",
           title="Calibration on CV out-of-fold predictions (seed 42)")
    style(ax)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths, ev, ccfg = config["paths"], config["evaluation"], config["calibration"]
    rng = np.random.default_rng(config["seed"])

    oof = pd.read_csv(paths["predictions_csv"], dtype={"case_id": str}).sort_values("case_id")
    classes = [c.removeprefix("prob_") for c in oof.columns if c.startswith("prob_")]
    probs = oof[[f"prob_{c}" for c in classes]].to_numpy()
    y_idx = oof.label.map(classes.index).to_numpy()
    low, high, n = ccfg["temperature_grid"]
    grid = np.geomspace(low, high, int(n))
    scaled, temps = cross_fitted_scaling(probs, y_idx, oof.cv_fold.to_numpy(), grid)
    n_bins = ccfg["n_bins"]

    results = with_ci({"raw": probs, "temperature_scaled": scaled}, y_idx, n_bins,
                      ev["n_bootstrap"], ev["ci_level"], rng)
    gap = results["raw"]["overconfidence_gap"]
    clearly = bool(gap["value"] >= ccfg["overconfidence_margin"] and gap["ci_low"] > 0)
    conf, correct = probs.max(axis=1), (probs.argmax(axis=1) == y_idx).astype(float)
    tables = {"raw": reliability_table(conf, correct, n_bins),
              "temperature_scaled": reliability_table(scaled.max(axis=1),
                                                      (scaled.argmax(axis=1) == y_idx).astype(float),
                                                      n_bins)}
    section = {
        "description": "Calibration of the seed-42 CV out-of-fold predictions (case level). "
                       "Temperature scaling is cross-fitted by CV fold. The test set was not used.",
        "config": args.config, "n_cases": int(len(oof)), "n_bins": n_bins,
        **results,
        "temperature_per_fold": temps,
        "temperature_fitted_on_all_cv": round(fit_temperature(probs, y_idx, grid), 3),
        "reliability_table": tables,
        "clearly_overconfident": clearly,
        "overconfidence_rule": f"mean confidence - accuracy >= {ccfg['overconfidence_margin']} "
                               "and its 95% CI excludes 0",
        "mean_predicted_vs_actual_share": {
            c: {"mean_predicted": round(float(probs[:, i].mean()), 4),
                "actual_share": round(float((y_idx == i).mean()), 4)}
            for i, c in enumerate(classes)},
    }
    metrics_path = Path(paths["metrics_json"])
    metrics = json.loads(metrics_path.read_text())
    metrics["calibration_cv"] = section
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    plot({"raw": tables["raw"], "temperature-scaled (cross-fitted)": tables["temperature_scaled"]},
         Path(paths["figures_dir"]) / "calibration_cv.png")
    print(json.dumps({k: section[k] for k in ("raw", "temperature_scaled",
                                               "temperature_scaled_minus_raw",
                                               "temperature_per_fold", "clearly_overconfident",
                                               "mean_predicted_vs_actual_share")}, indent=1))


if __name__ == "__main__":
    main()
