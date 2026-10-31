"""Phase 5: compare the image-processing settings with the baseline, on the same cases.

Reads, for every setting in `settings` (the first one is the baseline), the
out-of-fold CV predictions of each seed and the epoch budget it picked. Then,
overall and per eFST group:

- balanced accuracy, macro F1, macro AUC and the recall of the focus classes
  (psoriasis, tinea): mean over seeds, SD over seeds, 95% bootstrap CI;
- the paired change against the baseline: every setting is scored on the SAME
  bootstrap resample of cases, so case-to-case differences cancel out and only
  the effect of the setting remains. A change is "outside the noise" only if its
  95% CI excludes 0.

Finally the setting is chosen with the rule fixed in the config, on CV results
only. Everything goes to reports/metrics.json (`phase5_cv`).

Only CV-pool cases are used; the test set is never read.

Usage:
    python -m src.compare_settings --config configs/phase5/compare.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate import EFST_ORDER, METRICS, case_metrics, ci
from src.tone_analysis import load_predictions, r4, style
from src.utils import load_config

SUBSETS = ["overall"] + EFST_ORDER
BASELINE_COLOR = "#52514e"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua: validated all-pairs
MUTED = "#52514e"


def setting_predictions(setting_cfg):
    """load_predictions for one setting: fill {experiment} now, {seed} per seed."""
    paths = dict(setting_cfg["paths"])
    paths["predictions_csv"] = paths["predictions_csv"].format(
        experiment=setting_cfg["experiment"], seed="{seed}")
    return load_predictions({"seeds": setting_cfg["seeds"], "paths": paths})


def scores(y, probs, classes, focus):
    """Seed-mean metrics and focus-class recalls for one set of cases, plus per-seed values."""
    pred = np.asarray(classes)[probs.argmax(axis=2)]                   # seeds x cases
    per_seed = []
    for p, pr in zip(probs, pred):
        row = case_metrics(y, p, classes)
        for c in focus:
            row[f"{c}_recall"] = (pr[y == c] == c).mean() if (y == c).any() else np.nan
        per_seed.append(row)
    keys = list(per_seed[0])
    return {k: float(np.mean([r[k] for r in per_seed])) for k in keys}, per_seed


def summarise(point, per_seed, boot, key, ci_level):
    low, high = ci(boot, ci_level)
    values = [r[key] for r in per_seed]
    return {"mean": r4(point[key]), "sd_over_seeds": r4(np.std(values, ddof=1)),
            "per_seed": [r4(v) for v in values], "ci_low": low, "ci_high": high}


def compare_subset(y, probs_by_setting, classes, focus, ev, min_cell, rng):
    """Absolute scores per setting and paired changes against the first setting,
    all from one shared set of bootstrap resamples of these cases."""
    names = list(probs_by_setting)
    idx = rng.integers(0, len(y), size=(ev["n_bootstrap"], len(y)))
    points, per_seed, boots = {}, {}, {}
    for name, probs in probs_by_setting.items():
        points[name], per_seed[name] = scores(y, probs, classes, focus)
        samples = [scores(y[i], probs[:, i], classes, focus)[0] for i in idx]
        boots[name] = {k: np.array([s[k] for s in samples]) for k in points[name]}
    keys = list(points[names[0]])
    counts = {c: int((y == c).sum()) for c in focus}
    absolute = {}
    for name in names:
        absolute[name] = {"n_cases": int(len(y))}
        for k in keys:
            absolute[name][k] = summarise(points[name], per_seed[name], boots[name][k], k,
                                          ev["ci_level"])
            if k.endswith("_recall"):
                n = counts[k.removesuffix("_recall")]
                absolute[name][k].update(n_cases=n, reliable=n >= min_cell)
    base = names[0]
    paired = {}
    for name in names[1:]:
        paired[name] = {}
        for k in keys:
            low, high = ci(boots[name][k] - boots[base][k], ev["ci_level"])
            paired[name][k] = {"difference": r4(points[name][k] - points[base][k]),
                               "ci_low": low, "ci_high": high,
                               "outside_noise": bool(low > 0 or high < 0)}
            if k.endswith("_recall"):
                paired[name][k]["reliable"] = counts[k.removesuffix("_recall")] >= min_cell
    return absolute, paired


def select_setting(names, results, paired, metric):
    """The pre-registered rule: see `analysis` in configs/phase5/compare.yaml."""
    base, verdicts = names[0], {}
    for name in names[1:]:
        gain = paired["overall"][name][metric]
        worse = [g for g in EFST_ORDER if paired[g][name][metric]["ci_high"] < 0]
        verdicts[name] = {"overall_gain": gain["difference"],
                          "gain_outside_noise": gain["ci_low"] > 0,
                          "fst_groups_worse_outside_noise": worse,
                          "qualifies": bool(gain["ci_low"] > 0 and not worse)}
    qualifying = [n for n in names[1:] if verdicts[n]["qualifies"]]
    if qualifying:
        chosen = max(qualifying, key=lambda n: results["overall"][n][metric]["mean"])
        reason = (f"{chosen} improves overall {metric} over the baseline outside the noise "
                  "without making any FST group worse outside the noise"
                  + (" and has the highest mean of the settings that do" if len(qualifying) > 1
                     else ""))
    else:
        chosen = base
        reason = (f"no setting improves overall {metric} over the baseline outside the noise "
                  "(without hurting an FST group), so the simplest setting is kept")
    return {"selected": chosen, "reason": reason, "per_setting": verdicts}


# ---------- figures ----------

def setting_colors(names):
    return dict(zip(names, [BASELINE_COLOR] + COLORS))


def plot_absolute(section, path):
    names = section["settings"]
    colors = setting_colors(names)
    offsets = np.linspace(-0.27, 0.27, len(names))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 4.6), sharey=True)
    for ax, m in zip(axes, METRICS):
        for y, subset in enumerate(SUBSETS):
            for k, name in enumerate(names):
                v = section["results"][subset][name][m]
                ax.plot([v["ci_low"], v["ci_high"]], [y + offsets[k]] * 2, color=colors[name],
                        lw=2)
                ax.plot(v["mean"], y + offsets[k], "o", ms=7, color=colors[name],
                        label=name if y == 0 else None)
        ax.set(title=m.replace("_", " "), xlim=(0, 1.03))
        style(ax)
    n = {s: section["results"][s][names[0]]["n_cases"] for s in SUBSETS}
    axes[0].set_yticks(range(len(SUBSETS)), [f"{s} (n={n[s]})" for s in SUBSETS])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(names),
               fontsize=8, frameon=False)
    fig.suptitle(f"Phase 5 settings, {len(section['seeds'])} seeds (dot = seed mean, "
                 "line = 95% bootstrap CI)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def plot_paired(section, path):
    names = section["settings"]
    others = names[1:]
    colors = setting_colors(names)
    keys = METRICS + [f"{c}_recall" for c in section["focus_classes"]]
    offsets = np.linspace(-0.2, 0.2, len(others))
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 4.4), sharey=True)
    for ax, k in zip(axes, keys):
        for y, subset in enumerate(SUBSETS):
            for j, name in enumerate(others):
                d = section["paired_vs_baseline"][subset][name][k]
                hollow = d.get("reliable") is False
                ax.plot([100 * d["ci_low"], 100 * d["ci_high"]], [y + offsets[j]] * 2,
                        color=colors[name], lw=2, ls=":" if hollow else "-")
                ax.plot(100 * d["difference"], y + offsets[j], "o", ms=7, color=colors[name],
                        mfc="white" if hollow else colors[name],
                        label=f"{name} minus {names[0]}" if y == 0 else None)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set(title=k.replace("_", " "), xlabel="change (percentage points)")
        style(ax)
    axes[0].set_yticks(range(len(SUBSETS)), SUBSETS)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(others),
               fontsize=8, frameon=False)
    fig.suptitle("Paired change against the baseline on the same cases (95% bootstrap CI; "
                 "a CI crossing 0 is within noise; dotted / hollow = under "
                 f"{section['min_cell_cases']} cases)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def plot_budgets(section, path):
    names = section["settings"]
    colors = setting_colors(names)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in names:
        budget = section["epoch_budget"][name]
        if budget.get("source") != "budget search":
            continue
        scores_by_epoch = {int(e): v for e, v in budget["scores"].items()}  # JSON keys are text
        epochs = sorted(scores_by_epoch)
        ax.plot(epochs, [scores_by_epoch[e] for e in epochs], "-o", lw=2, ms=7,
                color=colors[name], label=name)
        ax.plot(budget["selected"], scores_by_epoch[budget["selected"]], "o", ms=13, mfc="none",
                mec=colors[name], mew=1.5)
    ax.set(xlabel="epoch budget", ylabel="balanced accuracy on fold 1 (seed 42)",
           title="Epoch budget search per setting (ring = chosen)")
    ax.set_xticks(epochs)
    style(ax)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths, ev, acfg = config["paths"], config["evaluation"], config["analysis"]
    rng = np.random.default_rng(config["seed"])
    focus = acfg["focus_classes"]

    setting_cfgs = {}
    for path in config["settings"]:
        cfg = load_config(path)
        setting_cfgs[cfg["experiment"]] = (path, cfg)
    names = list(setting_cfgs)

    cv, classes, probs = None, None, {}
    budgets = {}
    for name, (path, cfg) in setting_cfgs.items():
        cases, setting_classes, probs[name] = setting_predictions(cfg)
        assert cv is None or (cases.case_id.to_numpy() == cv.case_id.to_numpy()).all(), \
            "settings were evaluated on different cases"
        assert classes is None or setting_classes == classes, "settings disagree on classes"
        cv, classes = cases, setting_classes
        log_path = cfg["paths"]["train_log_json"].format(experiment=name, seed=cfg["seeds"][0])
        budgets[name] = json.loads(Path(log_path).read_text())["epoch_budget"]
    y = cv.label.to_numpy()

    results, paired = {}, {}
    for subset in SUBSETS:
        print(f"Bootstrapping {subset} ...", flush=True)
        mask = np.ones(len(cv), bool) if subset == "overall" else (cv.efst_group == subset).to_numpy()
        results[subset], paired[subset] = compare_subset(
            y[mask], {n: p[:, mask] for n, p in probs.items()}, classes, focus, ev,
            acfg["min_cell_cases"], rng)

    n_comparisons = sum(1 for s in paired for n in paired[s] for _ in paired[s][n])
    section = {
        "description": "Phase 5. Image-processing settings compared with the baseline on the "
                       "same CV cases (out-of-fold, case level, seed mean). The test set was "
                       "not used.",
        "config": args.config,
        "settings": names,
        "baseline": names[0],
        "setting_configs": {n: p for n, (p, _) in setting_cfgs.items()},
        "images": {n: {k: c["images"][k] for k in ("augmentation", "preprocessing")}
                   for n, (_, c) in setting_cfgs.items()},
        "seeds": setting_cfgs[names[0]][1]["seeds"],
        "focus_classes": focus,
        "min_cell_cases": acfg["min_cell_cases"],
        "epoch_budget": budgets,
        "results": results,
        "paired_vs_baseline": paired,
        "selection": select_setting(names, results, paired, acfg["selection_metric"]),
        "multiple_comparisons_note": (
            f"{n_comparisons} paired comparisons are reported. At a 95% level about 1 in 20 "
            "differences with no real effect will still look outside the noise, so a single "
            "isolated 'outside the noise' result, especially in a small group, is weak evidence."),
    }

    metrics_path = Path(paths["metrics_json"])
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["phase5_cv"] = section
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    figures = Path(paths["figures_dir"])
    plot_absolute(section, figures / "phase5_settings_by_fst.png")
    plot_paired(section, figures / "phase5_paired_vs_baseline.png")
    plot_budgets(section, figures / "phase5_epoch_budgets.png")
    print_summary(section)
    print(f"Wrote {metrics_path} (phase5_cv) and figures in {figures}")


def print_summary(section):
    names, base = section["settings"], section["baseline"]
    print("\nEpoch budget chosen on fold 1:", {n: b["selected"] for n, b in
                                               section["epoch_budget"].items()})
    print("Overall, seed mean [95% CI]:")
    for name in names:
        r = section["results"]["overall"][name]
        print(f"  {name:18s} " + "  ".join(
            f"{m} {r[m]['mean']:.3f} [{r[m]['ci_low']:.3f}, {r[m]['ci_high']:.3f}]" for m in METRICS))
    print(f"Paired change vs {base} (outside the noise marked *):")
    for subset in SUBSETS:
        for name in names[1:]:
            d = section["paired_vs_baseline"][subset][name]
            print(f"  {subset:10s} {name:18s} " + "  ".join(
                f"{k} {v['difference']:+.3f}{'*' if v['outside_noise'] else ' '}"
                for k, v in d.items()))
    sel = section["selection"]
    print(f"Selected: {sel['selected']} ({sel['reason']})")


if __name__ == "__main__":
    main()
