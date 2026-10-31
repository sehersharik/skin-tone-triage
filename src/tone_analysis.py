"""Phase 4: is the skin-tone gap real, or is it the mix of conditions?

Reads the out-of-fold CV predictions of every seed (from src.train with
configs/seeds.yaml) and, for each skin-tone grouping (eFST first, Monk US as a
secondary check):

1. Per-group metrics: balanced accuracy, macro F1 and macro AUC for each seed,
   reported as the mean over seeds, the spread over seeds (SD, min, max) and a
   95% bootstrap CI of the seed mean (cases resampled, same resample for every seed).
2. Focus-class recall: recall of dermatitis only, per group. Within one class the
   condition mix cannot differ between groups.
3. Accuracy, crude and class-mix standardised: the crude accuracy is the share
   of a group's cases predicted correctly. The standardised accuracy reweights
   each group to the same condition mix (the pooled CV mix):
       sum over classes c of  share_c (pooled) x recall_c (in this group)
   The bootstrap resamples within each group x class cell, so the mix stays fixed.
4. Logistic regression of "case predicted correctly" on tone group (+ condition,
   + photo factors), on the stacked rows of all seeds. The tone effect is an odds
   ratio against the lightest group, with a 95% CI from resampling cases.
5. Photo factors per group: brightness, contrast, sharpness and resolution.

"Correct" for a case is averaged over seeds where a single number per case is
needed (crude / standardised accuracy). All gaps count as "outside the noise"
only if their 95% CI excludes 0 (differences) or 1 (odds ratios).

Only CV-pool cases are used; the test set is never read.

Usage:
    python -m src.tone_analysis --config configs/seeds.yaml
"""
import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.evaluate import EFST_ORDER, METRICS, MONK_ORDER, case_metrics, ci
from src.utils import load_config

GROUPINGS = {"efst": ("efst_group", EFST_ORDER), "monk": ("monk_us_group", MONK_ORDER)}
FACTORS = ["brightness", "contrast", "sharpness", "megapixels"]
COLORS = ["#2a78d6", "#eb6834"]   # comparison 1 / comparison 2 (validated categorical pair)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def r4(x):
    return round(float(x), 4)


def load_predictions(config):
    """CV cases sorted by case_id, the class names, and probs[seed, case, class]."""
    paths = config["paths"]
    cases = pd.read_csv(paths["cases_csv"], dtype={"case_id": str})
    cv = cases[cases.split == "cv"].sort_values("case_id", ignore_index=True)
    probs, classes = [], None
    for seed in config["seeds"]:
        oof = pd.read_csv(paths["predictions_csv"].format(seed=seed), dtype={"case_id": str})
        assert oof.case_id.is_unique and set(oof.case_id) == set(cv.case_id), \
            f"seed {seed}: OOF predictions must cover exactly the CV pool"
        oof = oof.set_index("case_id").loc[cv.case_id]
        assert (oof.label.to_numpy() == cv.label.to_numpy()).all(), "labels differ from cases.csv"
        seed_classes = [c.removeprefix("prob_") for c in oof.columns if c.startswith("prob_")]
        assert classes is None or seed_classes == classes, "seeds disagree on the classes"
        classes = seed_classes
        probs.append(oof[[f"prob_{c}" for c in classes]].to_numpy())
    return cv, classes, np.stack(probs)


def pairwise_gaps(points, boots, order, ci_level):
    """`b minus a` for every pair of groups present, with bootstrap CIs. Group bootstraps
    are independent, so their samples can be subtracted element-wise."""
    out = {}
    for a, b in combinations([g for g in order if g in boots], 2):
        low, high = ci(boots[b] - boots[a], ci_level)
        out[f"{b} minus {a}"] = {"difference": r4(points[b] - points[a]), "ci_low": low,
                                 "ci_high": high, "outside_noise": bool(low > 0 or high < 0)}
    return out


# ---------- 1. per-group metrics across seeds ----------

def metrics_across_seeds(y, probs, classes, n_bootstrap, ci_level, rng):
    """case_metrics per seed; CI of the seed mean from resampling cases (paired across seeds)."""
    per_seed = pd.DataFrame([case_metrics(y, p, classes) for p in probs])
    idx = rng.integers(0, len(y), size=(n_bootstrap, len(y)))
    boot = pd.DataFrame([pd.DataFrame([case_metrics(y[i], p[i], classes) for p in probs]).mean()
                         for i in idx])
    out = {"n_cases": int(len(y))}
    for m in METRICS:
        low, high = ci(boot[m], ci_level)
        out[m] = {"mean": r4(per_seed[m].mean()), "sd_over_seeds": r4(per_seed[m].std(ddof=1)),
                  "min_seed": r4(per_seed[m].min()), "max_seed": r4(per_seed[m].max()),
                  "per_seed": [r4(v) for v in per_seed[m]], "ci_low": low, "ci_high": high}
    return out, boot


def by_group_across_seeds(cv, group_col, order, classes, probs, ev, rng):
    y = cv.label.to_numpy()
    results, boots = {}, {}
    for g in order:
        mask = (cv[group_col] == g).to_numpy()
        results[g], boots[g] = metrics_across_seeds(y[mask], probs[:, mask], classes,
                                                    ev["n_bootstrap"], ev["ci_level"], rng)
    gaps = {m: pairwise_gaps({g: results[g][m]["mean"] for g in order},
                             {g: boots[g][m].to_numpy() for g in order}, order, ev["ci_level"])
            for m in METRICS}
    return results, gaps


# ---------- 2. focus-class recall ----------

def class_recall(correct, y, mask, cls, n_bootstrap, rng):
    """Recall of one class within a group: per seed, and the seed mean with a case bootstrap."""
    c = correct[:, mask & (y == cls)]          # seeds x cases of that class in the group
    per_seed = c.mean(axis=1)
    per_case = c.mean(axis=0)
    boot = per_case[rng.integers(0, len(per_case), size=(n_bootstrap, len(per_case)))].mean(axis=1)
    return per_seed, boot


# ---------- 3. crude and standardised accuracy ----------

def standardised_accuracy(correct_case, y, mask, weights):
    """sum over classes of weights[c] x recall of c within `mask` (NaN if a class has no cases)."""
    return sum(w * correct_case[mask & (y == cls)].mean() if (mask & (y == cls)).any() else np.nan
               for cls, w in weights.items())


def standardised_bootstrap(correct_case, y, mask, weights, n_bootstrap, rng):
    """Resample inside each class cell of the group, so the condition mix stays fixed."""
    boot = np.zeros(n_bootstrap)
    for cls, w in weights.items():
        vals = correct_case[mask & (y == cls)]
        boot += w * vals[rng.integers(0, len(vals), size=(n_bootstrap, len(vals)))].mean(axis=1)
    return boot


def accuracy_section(cv, group_col, order, correct, ev, min_cell, rng):
    y = cv.label.to_numpy()
    labelled = cv[group_col].notna().to_numpy()
    weights = cv.loc[labelled, "label"].value_counts(normalize=True).sort_index().to_dict()
    crude, standard, boots_c, boots_s = {}, {}, {}, {}
    for g in order:
        mask = (cv[group_col] == g).to_numpy()
        per_case = correct[:, mask].mean(axis=0)
        boots_c[g] = per_case[rng.integers(0, len(per_case),
                                           size=(ev["n_bootstrap"], len(per_case)))].mean(axis=1)
        crude[g] = {"n_cases": int(mask.sum()), "value": r4(per_case.mean()),
                    "per_seed": [r4(v) for v in correct[:, mask].mean(axis=1)],
                    **dict(zip(("ci_low", "ci_high"), ci(boots_c[g], ev["ci_level"])))}
        cells = {c: int((mask & (y == c)).sum()) for c in weights}
        if min(cells.values()) == 0:
            standard[g] = {"n_cases": int(mask.sum()), "cases_per_class": cells,
                           "note": "a class has no cases, so this group cannot be standardised"}
            continue
        point = standardised_accuracy(correct.mean(axis=0), y, mask, weights)
        boots_s[g] = standardised_bootstrap(correct.mean(axis=0), y, mask, weights,
                                            ev["n_bootstrap"], rng)
        per_seed = [standardised_accuracy(c, y, mask, weights) for c in correct]
        small = [c for c, n in cells.items() if n < min_cell]
        standard[g] = {"n_cases": int(mask.sum()), "value": r4(point),
                       "per_seed": [r4(v) for v in per_seed],
                       **dict(zip(("ci_low", "ci_high"), ci(boots_s[g], ev["ci_level"]))),
                       "cases_per_class": cells, "small_cells": small,
                       "reliable": not small}
    std_points = {g: standard[g]["value"] for g in boots_s}
    std_gaps = pairwise_gaps(std_points, boots_s, order, ev["ci_level"])
    for name, gap in std_gaps.items():
        b, a = name.split(" minus ")
        gap["reliable"] = standard[a]["reliable"] and standard[b]["reliable"]
    return {
        "standard_mix": {c: r4(w) for c, w in weights.items()},
        "crude": crude,
        "crude_gaps": pairwise_gaps({g: crude[g]["value"] for g in order}, boots_c, order,
                                    ev["ci_level"]),
        "standardised": standard,
        "standardised_gaps": std_gaps,
        "note": f"Standardised CIs are too narrow when a cell has fewer than {min_cell} cases "
                "(the bootstrap cannot show how much such a small cell could vary); "
                "those groups are marked reliable: false.",
    }


# ---------- 4. logistic regression ----------

def design_matrix(df, group_col, order, classes, focus_class, factors=()):
    """One-hot tone group (reference = order[0]) and condition (reference = focus_class),
    plus z-scored photo factors."""
    X = pd.DataFrame(index=df.index)
    for g in order[1:]:
        X[f"tone: {g}"] = (df[group_col] == g).astype(float)
    for c in classes:
        if c != focus_class:
            X[f"condition: {c}"] = (df.label == c).astype(float)
    for f in factors:
        values = np.log(df[f]) if f in ("sharpness", "megapixels") else df[f]
        if values.std() > 0:  # a factor that never varies cannot explain anything
            X[f"photo: {f}"] = (values - values.mean()) / values.std()
    return X


def fit_coefs(X, correct):
    """Unpenalised logistic regression on the rows of every seed stacked; returns coefficients."""
    model = LogisticRegression(C=np.inf, max_iter=5000)  # C=inf: no penalty
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(np.tile(X, (correct.shape[0], 1)), correct.ravel())
    return model.coef_[0]


def regression_section(cv, group_col, order, classes, correct, focus_class, ev, rng):
    mask = cv[group_col].notna().to_numpy()
    if not all(f in cv for f in FACTORS):
        factor_sets = {"tone only": None, "tone + condition": ()}
    else:
        mask = mask & cv[FACTORS].notna().all(axis=1).to_numpy()
        factor_sets = {"tone only": None, "tone + condition": (),
                       "tone + condition + photo factors": FACTORS}
    df, c = cv[mask], correct[:, mask]
    tone_cols = [f"tone: {g}" for g in order[1:]]
    out = {"n_cases": int(mask.sum()), "n_rows": int(c.size), "reference_group": order[0],
           "outcome": "case predicted correctly (one row per case per seed)", "models": {}}
    idx = rng.integers(0, len(df), size=(ev["n_bootstrap"], len(df)))
    for name, factors in factor_sets.items():
        X = design_matrix(df, group_col, order, classes, focus_class, factors or ())
        if factors is None:
            X = X[tone_cols]
        X = X.to_numpy()
        point = fit_coefs(X, c)[:len(tone_cols)]
        per_seed = np.array([fit_coefs(X, c[[s]])[:len(tone_cols)] for s in range(len(c))])
        boot = np.array([fit_coefs(X[i], c[:, i])[:len(tone_cols)] for i in idx])
        model = {}
        for j, g in enumerate(order[1:]):
            low, high = ci(np.exp(boot[:, j]), ev["ci_level"])
            model[f"{g} vs {order[0]}"] = {
                "odds_ratio": r4(np.exp(point[j])), "ci_low": low, "ci_high": high,
                "outside_noise": bool(low > 1 or high < 1),
                "per_seed_odds_ratio": [r4(v) for v in np.exp(per_seed[:, j])]}
        out["models"][name] = model
    return out


# ---------- 5. photo factors ----------

def factor_section(cv, group_col, order, ev, rng):
    out = {}
    for f in FACTORS:
        points, boots, out[f] = {}, {}, {}
        for g in order:
            v = cv.loc[cv[group_col] == g, f].dropna().to_numpy()
            points[g] = v.mean()
            boots[g] = v[rng.integers(0, len(v), size=(ev["n_bootstrap"], len(v)))].mean(axis=1)
            low, high = ci(boots[g], ev["ci_level"])
            q25, q50, q75 = np.percentile(v, [25, 50, 75])
            out[f][g] = {"n_cases": int(len(v)), "mean": r4(v.mean()), "ci_low": low,
                         "ci_high": high, "median": r4(q50), "q25": r4(q25), "q75": r4(q75)}
        out[f]["gaps_in_mean"] = pairwise_gaps(points, boots, order, ev["ci_level"])
    return out


def case_factors(path):
    """Image-level factors -> one row per case (mean over the case's photos)."""
    images = pd.read_csv(path, dtype={"case_id": str})
    return images.groupby("case_id")[FACTORS].mean().reset_index()


# ---------- figures ----------

def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot_seeds(section, path):
    """Seed mean with 95% CI per group; hollow markers show each seed."""
    rows = [(g, "eFST", section["efst"]["metrics_by_group"][g]) for g in EFST_ORDER] + \
           [(g, "Monk (US)", section["monk"]["metrics_by_group"][g]) for g in MONK_ORDER]
    colors = {"eFST": COLORS[0], "Monk (US)": COLORS[1]}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 3.9), sharey=True)
    for ax, m in zip(axes, METRICS):
        for y, (group, kind, res) in enumerate(rows):
            v = res[m]
            ax.plot([v["ci_low"], v["ci_high"]], [y, y], color=colors[kind], lw=2)
            ax.plot(v["per_seed"], [y] * len(v["per_seed"]), "o", mfc="none", mec=colors[kind],
                    ms=8, mew=1)
            ax.plot(v["mean"], y, "o", color=colors[kind], ms=8,
                    label=kind if group in (EFST_ORDER[0], MONK_ORDER[0]) else None)
        ax.axvline(section["overall"][m]["mean"], color=MUTED, lw=1, ls="--")
        ax.set(title=m.replace("_", " "), xlim=(0, 1.03))
        style(ax)
    axes[0].set_yticks(range(len(rows)), [f"{g} (n={r['n_cases']})" for g, _, r in rows])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8,
               frameon=False)
    fig.suptitle(f"Baseline by skin-tone group, {len(section['seeds'])} seeds (filled = seed mean, "
                 "line = 95% bootstrap CI, hollow = each seed; dashed = overall)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def plot_gap_checks(part, order, focus_class, title, path):
    """Left: gaps against the lightest group, in percentage points, before and after
    accounting for condition mix. Right: tone odds ratios from the regressions."""
    ref = order[0]
    others = order[1:]
    measures = [("balanced accuracy", part["metrics_gaps"]["balanced_accuracy"]),
                ("accuracy, crude", part["accuracy"]["crude_gaps"]),
                ("accuracy, same condition mix", part["accuracy"]["standardised_gaps"]),
                (f"{focus_class} recall", part["focus_class_recall"]["gaps"])]
    models = list(part["regression"]["models"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2),
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    offsets = np.linspace(-0.15, 0.15, len(others))
    for k, g in enumerate(others):
        for y, (label, gaps) in enumerate(measures):
            gap = gaps.get(f"{g} minus {ref}")
            if gap is None:
                continue
            hollow = gap.get("reliable") is False
            left.plot([100 * gap["ci_low"], 100 * gap["ci_high"]], [y + offsets[k]] * 2,
                      color=COLORS[k], lw=2, ls=":" if hollow else "-")
            left.plot(100 * gap["difference"], y + offsets[k], "o", ms=8, color=COLORS[k],
                      mfc="white" if hollow else COLORS[k],
                      label=f"{g} minus {ref}" if y == 0 else None)
        for y, name in enumerate(models):
            res = part["regression"]["models"][name][f"{g} vs {ref}"]
            right.plot([res["ci_low"], res["ci_high"]], [y + offsets[k]] * 2, color=COLORS[k], lw=2)
            right.plot(res["odds_ratio"], y + offsets[k], "o", ms=8, color=COLORS[k])
    left.axvline(0, color=MUTED, lw=1)
    left.set_yticks(range(len(measures)), [m[0] for m in measures])
    left.set(xlabel="difference from " + ref + " (percentage points)",
             title="Gap before and after accounting for condition mix")
    right.axvline(1, color=MUTED, lw=1)
    right.set_xscale("log")
    right.minorticks_off()
    low, high = right.get_xlim()
    ticks = [t for t in (0.1, 0.25, 0.5, 1, 2, 4, 10) if low <= t <= high]
    right.set_xticks(ticks, [f"{t:g}" for t in ticks])
    right.set_yticks(range(len(models)), models)
    right.set(xlabel=f"odds ratio of a correct prediction\nvs {ref} (log scale)",
              title="Logistic regression: correct ~ ...")
    for ax in (left, right):
        ax.invert_yaxis()
        style(ax)
    fig.legend(*left.get_legend_handles_labels(), loc="lower center", ncol=len(others),
               fontsize=8, frameon=False)
    fig.suptitle(f"{title}: 95% bootstrap CIs; a CI crossing the grey line is within noise "
                 "(dotted / hollow = cells under 5 cases, CI too narrow)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def plot_factors(cv, path):
    groups = [g for g in EFST_ORDER if (cv.efst_group == g).any()]
    labels = {"brightness": "brightness (mean grey, 0-255)", "contrast": "contrast (SD of grey)",
              "sharpness": "sharpness (Laplacian variance, log)", "megapixels": "resolution (MP)"}
    fig, axes = plt.subplots(1, len(FACTORS), figsize=(13, 3.6))
    for ax, f in zip(axes, FACTORS):
        data = [cv.loc[cv.efst_group == g, f].dropna() for g in groups]
        ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#d6e6f8", "edgecolor": COLORS[0]},
                   medianprops={"color": INK}, whiskerprops={"color": MUTED},
                   capprops={"color": MUTED})
        ax.set_xticks(range(1, len(groups) + 1), [f"{g}\n(n={len(d)})" for g, d in zip(groups, data)],
                      fontsize=8)
        if f == "sharpness":
            ax.set_yscale("log")
        ax.set_title(labels[f], fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Photo factors by eFST group (one value per case = mean over its photos; "
                 "box = middle 50%, whiskers = 1.5 IQR)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths, ev, acfg = config["paths"], config["evaluation"], config["analysis"]
    rng = np.random.default_rng(config["seed"])

    cv, classes, probs = load_predictions(config)
    y = cv.label.to_numpy()
    correct = np.asarray(classes)[probs.argmax(axis=2)] == y        # seeds x cases
    factors_path = Path(paths["image_factors_csv"])
    if factors_path.exists():
        cv = cv.merge(case_factors(factors_path), on="case_id", how="left", validate="one_to_one")
    else:
        print(f"{factors_path} not found: run src.image_factors first; skipping photo factors")

    overall, _ = metrics_across_seeds(y, probs, classes, ev["n_bootstrap"], ev["ci_level"], rng)
    section = {
        "description": "Phase 4. Baseline EfficientNet-B0 (no augmentation, 5 epochs) rerun with "
                       "several seeds on the same CV folds; case-level out-of-fold predictions "
                       "on the CV pool. The test set was not used.",
        "config": args.config,
        "seeds": config["seeds"],
        "classes": classes,
        "overall": overall,
        "overall_accuracy_per_seed": [r4(v) for v in correct.mean(axis=1)],
    }
    focus = acfg["focus_class"]
    for key, (col, order) in GROUPINGS.items():
        print(f"Analysing {col} ...", flush=True)
        part = {"group_column": col}
        part["metrics_by_group"], part["metrics_gaps"] = by_group_across_seeds(
            cv, col, order, classes, probs, ev, rng)
        part["metrics_by_group"]["excluded_missing_group"] = int(cv[col].isna().sum())

        recall, boots, points = {}, {}, {}
        for g in order:
            mask = (cv[col] == g).to_numpy()
            per_seed, boots[g] = class_recall(correct, y, mask, focus, ev["n_bootstrap"], rng)
            points[g] = per_seed.mean()
            low, high = ci(boots[g], ev["ci_level"])
            recall[g] = {"n_cases": int((mask & (y == focus)).sum()), "mean": r4(points[g]),
                         "sd_over_seeds": r4(per_seed.std(ddof=1)),
                         "per_seed": [r4(v) for v in per_seed], "ci_low": low, "ci_high": high}
        part["focus_class_recall"] = {"class": focus, "by_group": recall,
                                      "gaps": pairwise_gaps(points, boots, order, ev["ci_level"])}
        # Point estimates only: most cells are too small for a useful CI. Shows which
        # classes drive a balanced-accuracy gap (balanced accuracy = mean of these recalls).
        part["per_class_recall_by_group"] = {
            g: {c: {"n_cases": int(((cv[col] == g).to_numpy() & (y == c)).sum()),
                    "recall": r4(correct[:, (cv[col] == g).to_numpy() & (y == c)].mean())}
                for c in classes} for g in order}
        part["accuracy"] = accuracy_section(cv, col, order, correct, ev, acfg["min_cell_cases"], rng)
        part["regression"] = regression_section(cv, col, order, classes, correct, focus, ev, rng)
        if all(f in cv for f in FACTORS):
            part["photo_factors"] = factor_section(cv, col, order, ev, rng)
        section[key] = part

    metrics_path = Path(paths["metrics_json"])
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["tone_analysis_cv"] = section
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    figures = Path(paths["figures_dir"])
    plot_seeds(section, figures / "phase4_by_skin_tone_seeds.png")
    plot_gap_checks(section["efst"], EFST_ORDER, focus, "eFST groups",
                    figures / "phase4_gap_checks_efst.png")
    plot_gap_checks(section["monk"], MONK_ORDER, focus, "Monk (US) groups",
                    figures / "phase4_gap_checks_monk.png")
    if "photo_factors" in section["efst"]:
        plot_factors(cv, figures / "phase4_photo_factors.png")
    print_summary(section)
    print(f"Wrote {metrics_path} (tone_analysis_cv) and figures in {figures}")


def print_summary(section):
    part = section["efst"]
    print("\neFST, balanced accuracy (seed mean [95% CI], SD over seeds):")
    for g in EFST_ORDER:
        v = part["metrics_by_group"][g]["balanced_accuracy"]
        print(f"  {g:11s} {v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]  SD {v['sd_over_seeds']:.3f}")
    print("Gaps vs FST I-II (difference [95% CI]):")
    rows = [("balanced accuracy", part["metrics_gaps"]["balanced_accuracy"]),
            ("accuracy, crude", part["accuracy"]["crude_gaps"]),
            ("accuracy, same mix", part["accuracy"]["standardised_gaps"]),
            (f"{part['focus_class_recall']['class']} recall", part["focus_class_recall"]["gaps"])]
    for label, gaps in rows:
        for name, gap in gaps.items():
            if name.endswith("minus FST I-II"):
                print(f"  {label:20s} {name:24s} {gap['difference']:+.3f} "
                      f"[{gap['ci_low']:+.3f}, {gap['ci_high']:+.3f}]")
    print("Odds ratio of a correct prediction vs FST I-II [95% CI]:")
    for model, res in part["regression"]["models"].items():
        for name, v in res.items():
            print(f"  {model:34s} {name:22s} {v['odds_ratio']:.2f} [{v['ci_low']:.2f}, {v['ci_high']:.2f}]")


if __name__ == "__main__":
    main()
