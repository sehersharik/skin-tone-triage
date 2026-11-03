"""Fill the README and the model card with numbers from reports/metrics.json.

The texts live in docs/templates/*.md with placeholders like {{test_ba}}. This
script computes every placeholder from metrics.json and writes README.md and
docs/MODEL_CARD.md, so no number in them is ever typed by hand. It stops if a
template uses a placeholder it doesn't know. A test checks the files are up to date.

Usage:
    python -m src.make_docs --config configs/app.yaml
"""
import argparse
import json
import re
from pathlib import Path

from src.app_logic import pretty
from src.evaluate import EFST_ORDER, MONK_ORDER
from src.utils import load_config

DOCS = {"docs/templates/README.md": "README.md",
        "docs/templates/MODEL_CARD.md": "docs/MODEL_CARD.md"}
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
METRIC_NAMES = {"balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1",
                "macro_auc_ovr": "Macro AUC"}


def num(x, digits=2):
    return f"{x:.{digits}f}"


def ci(m, key="value", digits=2):
    """0.45 (0.37–0.54)"""
    return f"{num(m[key], digits)} ({num(m['ci_low'], digits)}–{num(m['ci_high'], digits)})"


def pct(x):
    return f"{100 * x:.0f}%"


def pts(d, key="difference"):
    """+10.4 pts (+3.9 to +16.7), the CI of a difference in percentage points."""
    return f"{100 * d[key]:+.1f} pts ({100 * d['ci_low']:+.1f} to {100 * d['ci_high']:+.1f})"


def pct_range(values):
    low, high = pct(min(values)), pct(max(values))
    return f"about {low}" if low == high else f"{low}–{high}"


def star(d):
    return " *" if d["outside_noise"] else ""


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def values(m):
    v = {}
    cv, test, tone = m["baseline_cv"], m["test_final"], m["tone_analysis_cv"]
    p5, cam, cal, conf = m["phase5_cv"], m["phase6_gradcam"], m["calibration_cv"], m["app_confidence_cv"]
    classes = test["confusion_matrix"]["labels"]

    # Data
    v["n_classes"] = str(len(classes))
    v["class_list"] = ", ".join(pretty(c) for c in classes)
    v["chance"] = num(1 / len(classes))
    v["cv_cases"], v["test_cases"] = str(cv["overall"]["n_cases"]), str(test["n_cases"])
    v["cv_images"], v["test_images"] = str(cam["final_model"]["n_images"]), str(test["n_images"])
    v["n_seeds"] = str(len(tone["seeds"]))
    counts = [[pretty(c), cv["overall"]["cases_per_class"][c], test["overall"]["cases_per_class"][c]]
              for c in classes]
    v["class_counts_table"] = table(["Condition", "CV cases", "Test cases"], counts)

    # Headline numbers
    for name, sec in (("cv", cv["overall"]), ("test", test["overall"])):
        v[f"{name}_ba"] = ci(sec["balanced_accuracy"])
        v[f"{name}_ba_point"] = num(sec["balanced_accuracy"]["value"])
        v[f"{name}_auc"] = ci(sec["macro_auc_ovr"])
    comp = test["cv_comparison"]["metrics"]
    v["test_minus_cv_ba"] = pts(comp["balanced_accuracy"], "test_minus_cv")
    v["test_cv_agree"] = "within the noise" if not any(c["outside_noise"] for c in comp.values()) \
        else "outside the noise for at least one metric"

    rows = []
    for key, label in METRIC_NAMES.items():
        seeds = tone["overall"][key]
        rows.append([label, ci(cv["overall"][key]),
                     f"{num(seeds['mean'])} ± {num(seeds['sd_over_seeds'])} "
                     f"({num(seeds['ci_low'])}–{num(seeds['ci_high'])})",
                     ci(test["overall"][key]), pts(comp[key], "test_minus_cv") + star(comp[key])])
    v["overall_table"] = table(["Metric", "CV, seed 42", f"CV, {len(tone['seeds'])} seeds (mean ± SD)",
                                f"Test ({test['n_cases']} cases)", "Test minus CV"], rows)

    cm = test["confusion_matrix"]["rows_true_cols_pred"]
    v["test_recall_table"] = table(
        ["Condition", "Test cases", "Recognised", "Recall"],
        [[pretty(c), sum(cm[i]), cm[i][i], pct(cm[i][i] / sum(cm[i]))] for i, c in enumerate(classes)])

    # Per skin tone
    def tone_rows(by_cv, by_test, order, small):
        out = []
        for g in order:
            c, t = by_cv[g], by_test[g]
            flag = " (weak evidence)" if small[g]["weak_evidence"] else ""
            out.append([g, c["n_cases"], f"{num(c['balanced_accuracy']['mean'])} "
                        f"({num(c['balanced_accuracy']['ci_low'])}–{num(c['balanced_accuracy']['ci_high'])})",
                        t["n_cases"], ci(t["balanced_accuracy"]) + flag,
                        ci(t["macro_auc_ovr"]) if t["macro_auc_ovr"]["value"] == t["macro_auc_ovr"]["value"]
                        else "can't be computed"])
        return out
    header = ["Group", "CV cases", f"CV balanced accuracy ({len(tone['seeds'])}-seed mean)",
              "Test cases", "Test balanced accuracy", "Test AUC"]
    v["fst_table"] = table(header, tone_rows(tone["efst"]["metrics_by_group"], test["by_efst_group"],
                                             EFST_ORDER, test["small_groups"]["efst"]))
    v["monk_table"] = table(header, tone_rows(tone["monk"]["metrics_by_group"], test["by_monk_group"],
                                              MONK_ORDER, test["small_groups"]["monk"]))
    dark = test["by_efst_group"]["FST V-VI"]
    v["fst56_test_n"] = str(dark["n_cases"])
    v["fst56_test_ba"] = ci(dark["balanced_accuracy"])
    v["fst56_absent"] = ", ".join(pretty(c) for c in test["small_groups"]["efst"]["FST V-VI"]["classes_absent"])
    v["monk710_test_n"] = str(test["by_monk_group"]["Monk 7-10"]["n_cases"])
    v["fst56_cv_n"] = str(tone["efst"]["metrics_by_group"]["FST V-VI"]["n_cases"])

    # Phase 4
    e = tone["efst"]
    gap = "FST III-IV minus FST I-II"
    v["p4_ba_gap"] = pts(e["metrics_gaps"]["balanced_accuracy"][gap]) + star(e["metrics_gaps"]["balanced_accuracy"][gap])
    v["p4_crude_gap"] = pts(e["accuracy"]["crude_gaps"][gap]) + star(e["accuracy"]["crude_gaps"][gap])
    v["p4_std_gap"] = pts(e["accuracy"]["standardised_gaps"][gap]) + star(e["accuracy"]["standardised_gaps"][gap])
    v["p4_derm_gap"] = pts(e["focus_class_recall"]["gaps"][gap]) + star(e["focus_class_recall"]["gaps"][gap])
    odds = e["regression"]["models"]["tone + condition"]["FST III-IV vs FST I-II"]
    v["p4_or"] = f"{num(odds['odds_ratio'])} ({num(odds['ci_low'])}–{num(odds['ci_high'])})" + star(odds)
    odds_photo = e["regression"]["models"]["tone + condition + photo factors"]["FST III-IV vs FST I-II"]
    v["p4_or_photo"] = f"{num(odds_photo['odds_ratio'])} ({num(odds_photo['ci_low'])}–{num(odds_photo['ci_high'])})"
    rec = e["per_class_recall_by_group"]
    for c in ("psoriasis", "tinea"):
        for g, key in (("FST I-II", "12"), ("FST III-IV", "34")):
            v[f"p4_{c}_{key}"] = f"{pct(rec[g][c]['recall'])} of {rec[g][c]['n_cases']}"
    test_gap = test["efst_gaps"][gap]["balanced_accuracy"]
    v["test_fst_gap"] = pts(test_gap) + star(test_gap)

    # Phase 5
    names = p5["settings"]
    labels = {"p5_none": "Baseline (none)", "p5_standard_aug": "Standard augmentation",
              "p5_shades_of_gray": "Shades of Gray", "p5_clahe": "CLAHE"}
    rows = []
    for n in names:
        r = p5["results"]["overall"][n]
        if n == p5["baseline"]:
            deltas = ["—"] * 4
        else:
            d, d12 = p5["paired_vs_baseline"]["overall"][n], p5["paired_vs_baseline"]["FST I-II"][n]
            deltas = [pts(d["balanced_accuracy"]) + star(d["balanced_accuracy"]),
                      pts(d["macro_f1"]) + star(d["macro_f1"]),
                      pts(d["macro_auc_ovr"]) + star(d["macro_auc_ovr"]),
                      pts(d12["psoriasis_recall"]) + star(d12["psoriasis_recall"])]
        rows.append([labels.get(n, n), p5["epoch_budget"][n]["selected"],
                     f"{num(r['balanced_accuracy']['mean'])} ({num(r['balanced_accuracy']['ci_low'])}–"
                     f"{num(r['balanced_accuracy']['ci_high'])})", *deltas])
    v["p5_table"] = table(["Setting", "Epochs (picked on fold 1)", "Balanced accuracy (seed mean)",
                           "Δ balanced accuracy", "Δ macro F1", "Δ macro AUC",
                           "Δ psoriasis recall, FST I-II"], rows)
    v["aug_epochs"] = str(p5["epoch_budget"]["p5_standard_aug"]["selected"])
    v["base_epochs"] = str(p5["epoch_budget"][p5["baseline"]]["selected"])
    v["p5_selected"] = labels.get(p5["selection"]["selected"], p5["selection"]["selected"])
    v["p5_n_comparisons"] = re.search(r"\d+", p5["multiple_comparisons_note"]).group()

    # Grad-CAM
    b = cam["border_share_by_group"]
    v["cam_border_all"] = f"{pct(b['all']['mean'])} ({pct(b['all']['ci_low'])}–{pct(b['all']['ci_high'])})"
    v["cam_border_uniform"] = pct(cam["border_share_if_uniform"])
    correct = [b[f"{g}, correct"]["mean"] for g in EFST_ORDER]
    wrong = [b[f"{g}, incorrect"]["mean"] for g in EFST_ORDER]
    v["cam_border_correct"] = pct_range(correct)
    v["cam_border_incorrect"] = pct_range(wrong)
    v["cam_n_images"] = str(cam["n_images"])

    # Calibration and the app's "uncertain" rule
    raw = cal["raw"]
    v["cal_ece"] = ci(raw["ece"], digits=3)
    v["cal_gap"] = f"{100 * raw['overconfidence_gap']['value']:+.1f} pts " \
                   f"({100 * raw['overconfidence_gap']['ci_low']:+.1f} to {100 * raw['overconfidence_gap']['ci_high']:+.1f})"
    high = [r for r in cal["reliability_table"]["raw"] if r["n_cases"] and float(r["bin"].split("-")[0]) >= 0.8]
    n_high = sum(r["n_cases"] for r in high)
    v["cal_high_n"] = str(n_high)
    v["cal_high_conf"] = pct(sum(r["mean_confidence"] * r["n_cases"] for r in high) / n_high)
    v["cal_high_acc"] = pct(sum(r["accuracy"] * r["n_cases"] for r in high) / n_high)
    temps = cal["temperature_per_fold"].values()
    v["cal_temps"] = f"{num(min(temps))}–{num(max(temps))}"
    d = cal["temperature_scaled_minus_raw"]["ece"]
    v["cal_temp_ece"] = f"{d['difference']:+.3f} ({d['ci_low']:+.3f} to {d['ci_high']:+.3f})"
    share = cal["mean_predicted_vs_actual_share"]["dermatitis"]
    v["cal_derm_pred"], v["cal_derm_share"] = pct(share["mean_predicted"]), pct(share["actual_share"])
    v["app_threshold"] = pct(conf["threshold"])
    v["app_share_uncertain"] = pct(conf["share_uncertain"])
    v["app_acc_confident"] = pct(conf["accuracy_when_confident"]["value"])
    v["app_acc_uncertain"] = pct(conf["accuracy_when_uncertain"]["value"])

    # Model file
    v["model_sha256"] = test["model_sha256"]
    fm = cam["final_model"]
    v["model_epochs"], v["model_seed"] = str(fm["epochs"]), str(fm["seed"])
    v["model_torch"] = fm["torch_version"]
    return v


def render(template, v, name):
    unknown = sorted(set(PLACEHOLDER.findall(template)) - set(v))
    if unknown:
        raise SystemExit(f"{name}: unknown placeholders {unknown}")
    return PLACEHOLDER.sub(lambda match: v[match.group(1)], template)


def build(metrics_path):
    v = values(json.loads(Path(metrics_path).read_text()))
    return {out: render(Path(src).read_text(), v, src) for src, out in DOCS.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    for out, text in build(load_config(args.config)["paths"]["metrics_json"]).items():
        Path(out).write_text(text)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
