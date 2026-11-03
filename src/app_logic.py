"""Logic behind the Streamlit app, kept apart from the page so it can be tested.

- `reliability_numbers` reads every performance number the app shows from
  reports/metrics.json; nothing is hardcoded. The note is the same for every
  upload: the app does not try to judge the user's skin tone.
- `prepare_image` repeats the training pipeline on an uploaded photo (fix the
  phone's rotation, resize the short side to 256 px like the training cache,
  then the evaluation transform), entirely in memory.
- `confidence_state` and `format_probabilities` decide what the page says.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.prepare_images import resize_short_side

PRETTY = {"dermatitis": "Dermatitis (eczema)", "folliculitis": "Folliculitis",
          "insect_bite": "Insect bite", "psoriasis": "Psoriasis", "tinea": "Tinea (ringworm)",
          "urticaria": "Urticaria (hives)"}
REQUIRED_SECTIONS = ("baseline_cv", "test_final", "app_confidence_cv")


def pretty(name):
    return PRETTY.get(name, name.replace("_", " ").capitalize())


def format_probabilities(probs, classes):
    """[(pretty class name, probability, "42.1%")], highest first. All classes, always."""
    order = np.argsort(-np.asarray(probs), kind="stable")
    return [(pretty(classes[i]), float(probs[i]), f"{100 * probs[i]:.1f}%") for i in order]


def confidence_state(probs, threshold):
    """"uncertain" if the top probability is below the threshold, else "confident"."""
    return "uncertain" if float(np.max(probs)) < threshold else "confident"


def prepare_image(file_or_image, cache_short_side):
    """Uploaded photo -> RGB PIL image prepared like the training images (no file written)."""
    image = file_or_image if isinstance(file_or_image, Image.Image) else Image.open(file_or_image)
    image = ImageOps.exif_transpose(image).convert("RGB")      # phones store rotation in EXIF
    return resize_short_side(image, cache_short_side)


def _metric(section, name):
    m = section[name]
    return {"value": m["value"], "ci_low": m["ci_low"], "ci_high": m["ci_high"]}


def reliability_numbers(metrics_path):
    """Everything the reliability note shows, straight from metrics.json."""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    metrics = json.loads(path.read_text())
    missing = [s for s in REQUIRED_SECTIONS if s not in metrics]
    if missing:
        raise KeyError(f"{path} has no {', '.join(missing)} section")
    test, cv, conf = metrics["test_final"], metrics["baseline_cv"], metrics["app_confidence_cv"]
    labels = test["confusion_matrix"]["labels"]
    cm = np.asarray(test["confusion_matrix"]["rows_true_cols_pred"])
    dark_fst, dark_monk = test["by_efst_group"]["FST V-VI"], test["by_monk_group"]["Monk 7-10"]
    return {
        "test": {"n_cases": test["n_cases"],
                 "balanced_accuracy": _metric(test["overall"], "balanced_accuracy"),
                 "macro_auc_ovr": _metric(test["overall"], "macro_auc_ovr")},
        "cv": {"n_cases": cv["overall"]["n_cases"],
               "balanced_accuracy": _metric(cv["overall"], "balanced_accuracy"),
               "macro_auc_ovr": _metric(cv["overall"], "macro_auc_ovr")},
        "test_recall_by_class": [{"class": pretty(c), "correct": int(cm[i, i]),
                                  "cases": int(cm[i].sum())} for i, c in enumerate(labels)],
        "fst_v_vi": {"n_cases": dark_fst["n_cases"],
                     "classes_absent": [pretty(c) for c in
                                        test["small_groups"]["efst"]["FST V-VI"]["classes_absent"]],
                     "balanced_accuracy": _metric(dark_fst, "balanced_accuracy")},
        "monk_7_10": {"n_cases": dark_monk["n_cases"]},
        "confidence": {"threshold": conf["threshold"], "share_uncertain": conf["share_uncertain"],
                       "accuracy_when_confident": conf["accuracy_when_confident"],
                       "accuracy_when_uncertain": conf["accuracy_when_uncertain"]},
        "model_sha256": test["model_sha256"],
        "classes": labels,
    }


def ci_text(m, ci_level, digits=2):
    return (f"{m['value']:.{digits}f} ({ci_level:.0%} CI "
            f"{m['ci_low']:.{digits}f}–{m['ci_high']:.{digits}f})")


def pct_text(m, ci_level):
    return f"{m['value']:.0%} ({ci_level:.0%} CI {m['ci_low']:.0%}–{m['ci_high']:.0%})"
