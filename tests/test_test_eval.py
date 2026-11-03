"""Phase 7 checks: the test-set guard, and that scoring the test set twice gives the same numbers.

The end-to-end test runs src.test_eval on a small synthetic dataset, never on the real test set.
The last test re-scores the SAVED real test predictions (no model, no images) and checks they
reproduce reports/metrics.json exactly; it is skipped until the real evaluation has run.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

torch = pytest.importorskip("torch")

from src.data import split_fingerprint  # noqa: E402
from src.models import build_model  # noqa: E402
from src.test_eval import PROVENANCE, SECTION, check_not_done, load_test_set, score  # noqa: E402
from src.utils import load_config  # noqa: E402

CLASSES = ["dermatitis", "folliculitis", "insect_bite", "psoriasis", "tinea", "urticaria"]


def test_guard_refuses_a_second_scoring(tmp_path):
    paths = {"metrics_json": tmp_path / "metrics.json",
             "test_predictions_csv": tmp_path / "test_preds.csv"}
    check_not_done(paths, overwrite=False)                     # nothing yet: fine
    paths["metrics_json"].write_text(json.dumps({SECTION: {"overall": {}}}))
    with pytest.raises(SystemExit, match="already been scored"):
        check_not_done(paths, overwrite=False)
    check_not_done(paths, overwrite=True)                      # the owner's explicit override
    paths["metrics_json"].write_text("{}")
    paths["test_predictions_csv"].write_text("case_id\n")
    with pytest.raises(SystemExit):
        check_not_done(paths, overwrite=False)


def synthetic_project(tmp_path):
    """A tiny cases.csv (CV + test), cached images, an untrained model file and CV predictions."""
    rng = np.random.default_rng(0)
    (tmp_path / "cache").mkdir()
    rows = []
    for i in range(36):
        name = f"img_{i}.png"
        Image.fromarray((rng.random((40, 32, 3)) * 255).astype("uint8")).save(tmp_path / "cache" / name)
        rows.append({"case_id": f"{i:03d}", "label": CLASSES[i % 6],
                     "split": "test" if i < 18 else "cv", "cv_fold": i % 5,
                     "image_1_path": f"dataset/images/{name}", "image_2_path": None,
                     "image_3_path": None,
                     "efst_group": ["FST I-II", "FST III-IV", "FST V-VI"][i % 3],
                     "monk_us_group": ["Monk 1-3", "Monk 4-6"][i % 2]})
    cases = pd.DataFrame(rows)
    cases.to_csv(tmp_path / "cases.csv", index=False)
    (tmp_path / "fp.txt").write_text(split_fingerprint(
        pd.read_csv(tmp_path / "cases.csv", dtype={"case_id": str})))
    torch.manual_seed(0)
    torch.save(build_model("efficientnet_b0", 6, pretrained=False).state_dict(), tmp_path / "m.pt")
    (tmp_path / "m.json").write_text(json.dumps(
        {"classes": CLASSES, "seed": 42, "epochs": 5, "n_cases": 18, "n_images": 18,
         "torch_version": torch.__version__}))
    cv = cases[cases.split == "cv"]
    probs = rng.dirichlet(np.ones(6), len(cv))
    cv_preds = pd.DataFrame(probs, columns=[f"prob_{c}" for c in CLASSES])
    cv_preds["case_id"], cv_preds["label"] = cv.case_id.to_numpy(), cv.label.to_numpy()
    cv_preds.to_csv(tmp_path / "cv_preds.csv", index=False)
    config = {"base": "configs/test.yaml", "paths": {
        "cases_csv": str(tmp_path / "cases.csv"), "split_fingerprint": str(tmp_path / "fp.txt"),
        "image_cache_dir": str(tmp_path / "cache"), "runs_dir": str(tmp_path / "runs"),
        "final_model": str(tmp_path / "m.pt"), "final_model_info": str(tmp_path / "m.json"),
        "cv_predictions_csv": str(tmp_path / "cv_preds.csv"),
        "test_predictions_csv": str(tmp_path / "test_preds.csv"),
        "metrics_json": str(tmp_path / "metrics.json"), "figures_dir": str(tmp_path)},
        "images": {"input_size": 32}, "evaluation": {"n_bootstrap": 50}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    return tmp_path / "config.yaml"


def run_test_eval(config_path, *flags):
    return subprocess.run([sys.executable, "-m", "src.test_eval", "--config", str(config_path),
                           *flags], capture_output=True, text=True)


def numbers(metrics_path):
    section = json.loads(Path(metrics_path).read_text())[SECTION]
    return {k: v for k, v in section.items() if k not in PROVENANCE}


def test_scoring_twice_gives_identical_numbers_and_needs_the_override(tmp_path):
    config_path = synthetic_project(tmp_path)
    first = run_test_eval(config_path)
    assert first.returncode == 0, first.stderr
    result_1 = numbers(tmp_path / "metrics.json")
    assert result_1["n_cases"] == 18                            # test cases only

    blocked = run_test_eval(config_path)                       # no override -> refused
    assert blocked.returncode != 0 and "already been scored" in blocked.stderr + blocked.stdout
    assert numbers(tmp_path / "metrics.json") == result_1      # and nothing was rewritten

    second = run_test_eval(config_path, "--overwrite-test-result")
    assert second.returncode == 0, second.stderr
    assert numbers(tmp_path / "metrics.json") == result_1      # identical numbers


def test_load_test_set_returns_test_cases_only(tmp_path):
    synthetic_project(tmp_path)
    test = load_test_set({"cases_csv": tmp_path / "cases.csv",
                          "split_fingerprint": tmp_path / "fp.txt"})
    assert (test.split == "test").all() and len(test) == 18


CONFIG = load_config("configs/test.yaml")


@pytest.mark.skipif(not Path(CONFIG["paths"]["test_predictions_csv"]).exists(),
                    reason="the test set has not been scored yet")
def test_saved_test_result_reproduces_from_saved_predictions():
    paths = CONFIG["paths"]
    stored = json.loads(Path(paths["metrics_json"]).read_text())[SECTION]
    preds = pd.read_csv(paths["test_predictions_csv"], dtype={"case_id": str})
    cases = load_test_set(paths)
    cv_preds = pd.read_csv(paths["cv_predictions_csv"], dtype={"case_id": str})
    again = score(preds, cases, cv_preds, stored["confusion_matrix"]["labels"],
                  CONFIG["evaluation"], CONFIG["seed"], CONFIG["test"]["small_group_cases"])
    assert json.loads(json.dumps(again)) == {k: stored[k] for k in again}
