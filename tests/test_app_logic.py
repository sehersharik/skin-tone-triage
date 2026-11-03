"""App checks: probability formatting, the low-confidence rule, reading metrics.json, image
preparation, no hardcoded numbers, and the page itself (banner, note, missing weights)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.app_confidence import confidence_summary
from src.app_logic import (confidence_state, ensure_weights, format_probabilities,
                           prepare_image, reliability_numbers, sha256_of)
from src.utils import load_config

CLASSES = ["dermatitis", "folliculitis", "insect_bite", "psoriasis", "tinea", "urticaria"]
APP_CONFIG = load_config("configs/app.yaml")
METRICS = Path(APP_CONFIG["paths"]["metrics_json"])


def test_format_probabilities_lists_all_classes_highest_first():
    probs = np.array([0.1, 0.05, 0.421, 0.2, 0.029, 0.2])
    rows = format_probabilities(probs, CLASSES)
    assert len(rows) == 6
    assert rows[0] == ("Insect bite", pytest.approx(0.421), "42.1%")
    assert [p for _, p, _ in rows] == sorted(probs, reverse=True)
    assert rows[1][0] == "Psoriasis" and rows[2][0] == "Urticaria (hives)"   # ties keep class order
    assert rows[-1][2] == "2.9%"


def test_low_confidence_rule():
    assert confidence_state([0.5, 0.3, 0.2], 0.5) == "confident"            # at the threshold
    assert confidence_state([0.4999, 0.3, 0.2001], 0.5) == "uncertain"
    assert confidence_state(np.full(6, 1 / 6), 0.5) == "uncertain"


def test_confidence_summary():
    top = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 0, 0, 0])
    s = confidence_summary(top, correct, 0.5, 100, 0.95, np.random.default_rng(0))
    assert s["share_uncertain"] == 0.5
    assert s["accuracy_when_confident"]["value"] == 0.5
    assert s["accuracy_when_uncertain"]["value"] == 0.0


def fake_metrics():
    m = lambda v: {"value": v, "ci_low": v - 0.1, "ci_high": v + 0.1}  # noqa: E731
    group = {"n_cases": 15, "balanced_accuracy": m(0.2)}
    return {
        "baseline_cv": {"overall": {"n_cases": 928, "balanced_accuracy": m(0.38),
                                    "macro_auc_ovr": m(0.75)}},
        "test_final": {"n_cases": 233, "overall": {"balanced_accuracy": m(0.45),
                                                   "macro_auc_ovr": m(0.79)},
                       "confusion_matrix": {"labels": CLASSES,
                                            "rows_true_cols_pred": np.eye(6, dtype=int).tolist()},
                       "by_efst_group": {"FST V-VI": group},
                       "by_monk_group": {"Monk 7-10": {"n_cases": 3}},
                       "small_groups": {"efst": {"FST V-VI": {"classes_absent": ["folliculitis"]}}},
                       "model_sha256": "abc"},
        "app_confidence_cv": {"threshold": 0.5, "share_uncertain": 0.58,
                              "accuracy_when_confident": m(0.58),
                              "accuracy_when_uncertain": m(0.38)},
    }


def test_reliability_numbers_come_from_the_file(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(fake_metrics()))
    r = reliability_numbers(path)
    assert r["test"]["balanced_accuracy"]["value"] == 0.45
    assert r["fst_v_vi"] == {"n_cases": 15, "classes_absent": ["Folliculitis"],
                             "balanced_accuracy": {"value": 0.2, "ci_low": 0.1, "ci_high": 0.30000000000000004}}
    assert r["monk_7_10"]["n_cases"] == 3
    assert r["test_recall_by_class"][0] == {"class": "Dermatitis (eczema)", "correct": 1, "cases": 1}


def test_reliability_numbers_refuse_an_incomplete_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        reliability_numbers(tmp_path / "missing.json")
    incomplete = fake_metrics()
    del incomplete["test_final"]
    (tmp_path / "m.json").write_text(json.dumps(incomplete))
    with pytest.raises(KeyError, match="test_final"):
        reliability_numbers(tmp_path / "m.json")


@pytest.mark.skipif(not METRICS.exists(), reason="no metrics.json")
def test_real_metrics_match_what_the_app_reads():
    metrics = json.loads(METRICS.read_text())
    r = reliability_numbers(METRICS)
    test = metrics["test_final"]
    assert r["test"]["balanced_accuracy"]["value"] == test["overall"]["balanced_accuracy"]["value"]
    assert r["fst_v_vi"]["n_cases"] == test["by_efst_group"]["FST V-VI"]["n_cases"]
    assert r["cv"]["macro_auc_ovr"]["value"] == metrics["baseline_cv"]["overall"]["macro_auc_ovr"]["value"]


def displayed_text(path):
    """The literal text of a module's strings, without docstrings and f-string format codes."""
    import ast
    tree = ast.parse(Path(path).read_text())
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef)) and ast.get_docstring(node):
            skip.add(id(node.body[0].value))
        if isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            skip.update(id(n) for n in ast.walk(node.format_spec))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip]


def test_app_code_has_no_hardcoded_numbers():
    """Numbers like 0.45, 45% or "6 conditions" must come from metrics.json or the config."""
    for path in ("app/streamlit_app.py", "src/app_logic.py"):
        for text in displayed_text(path):
            assert not re.search(r"\d+\.\d+|\d+\s?%|\b\d+ (conditions|classes|cases)\b", text), \
                f"{path}: {text!r}"


def test_prepare_image_fixes_phone_rotation_and_resizes():
    image = Image.new("RGB", (400, 300), "white")
    exif = image.getexif()
    exif[0x0112] = 6                                   # EXIF orientation: rotate 90 degrees
    rotated = Image.open(_to_bytes(image, exif))
    out = prepare_image(rotated, 256)
    assert out.size == (256, 341)                       # upright (portrait), short side 256
    assert out.mode == "RGB"


def _to_bytes(image, exif):
    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return buf


@pytest.mark.skipif(not METRICS.exists(), reason="no metrics.json")
def test_page_shows_banner_note_and_handles_missing_weights(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    config = {"base": "configs/app.yaml", "paths": {"final_model": str(tmp_path / "none.pt")}}
    (tmp_path / "app.yaml").write_text(json.dumps(config))     # JSON is valid YAML
    monkeypatch.setattr("sys.argv", ["streamlit_app.py", "--config", str(tmp_path / "app.yaml")])
    at = testing.AppTest.from_file(str(Path("app/streamlit_app.py").resolve()),
                                        default_timeout=60).run()
    assert not at.exception
    assert "Educational demo, not a medical diagnosis. Do not use to make health decisions." \
        in at.error[0].value
    warnings = " ".join(w.value for w in at.warning)
    assert "are not on this machine" in warnings
    assert "Darker skin is poorly tested" in warnings


def test_ensure_weights_downloads_only_the_tested_file(tmp_path):
    source = tmp_path / "release" / "final_baseline.pt"
    source.parent.mkdir()
    source.write_bytes(b"pretend weights")
    url, good = source.as_uri(), sha256_of(source)          # file:// URL: no network needed
    target = tmp_path / "models" / "final_baseline.pt"

    assert ensure_weights(target, None, good) == "missing"   # inactive: no URL configured
    assert not target.exists()
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_weights(target, url, "0" * 64)                 # wrong fingerprint -> rejected
    assert not target.exists() and not list(target.parent.glob("*.download"))
    assert ensure_weights(target, url, good) == "downloaded"
    assert target.read_bytes() == b"pretend weights"
    assert ensure_weights(target, url, good) == "present"     # no second download


def test_app_config_keeps_the_download_inactive():
    assert APP_CONFIG["app"]["weights_url"] is None
