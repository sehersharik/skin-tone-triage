"""Measure the app's low-confidence rule on the CV out-of-fold predictions.

For the threshold in `app.low_confidence_threshold`: what share of CV cases fall
below it ("uncertain" in the app), and how often the top guess is right above and
below it, with 95% bootstrap CIs (cases resampled). Uses the seed-42 out-of-fold
predictions only; the test set is not used. Writes `app_confidence_cv` to
reports/metrics.json, which the app reads.

Usage:
    python -m src.app_confidence --config configs/app.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import ci
from src.utils import load_config


def confidence_summary(top_prob, correct, threshold, n_bootstrap, ci_level, rng):
    top_prob, correct = np.asarray(top_prob), np.asarray(correct, dtype=float)
    above = top_prob >= threshold
    out = {"threshold": threshold, "n_cases": int(len(correct)),
           "share_uncertain": round(float((~above).mean()), 4)}
    for name, mask in (("accuracy_when_confident", above), ("accuracy_when_uncertain", ~above)):
        vals = correct[mask]
        boot = vals[rng.integers(0, len(vals), size=(n_bootstrap, len(vals)))].mean(axis=1)
        low, high = ci(boot, ci_level)
        out[name] = {"n_cases": int(mask.sum()), "value": round(float(vals.mean()), 4),
                     "ci_low": low, "ci_high": high}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths, ev = config["paths"], config["evaluation"]
    oof = pd.read_csv(paths["predictions_csv"], dtype={"case_id": str})
    probs = oof[[c for c in oof.columns if c.startswith("prob_")]]
    section = confidence_summary(probs.max(axis=1), oof.pred == oof.label,
                                 config["app"]["low_confidence_threshold"], ev["n_bootstrap"],
                                 ev["ci_level"], np.random.default_rng(config["seed"]))
    section = {"description": "The app's low-confidence rule on the seed-42 CV out-of-fold "
                              "predictions (case level). The test set was not used.",
               "config": args.config, **section}
    metrics_path = Path(paths["metrics_json"])
    metrics = json.loads(metrics_path.read_text())
    metrics["app_confidence_cv"] = section
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(section, indent=2))


if __name__ == "__main__":
    main()
