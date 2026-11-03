"""The README and model card match metrics.json; the calibration measures behave as expected."""
from pathlib import Path

import numpy as np
import pytest

from src.calibration import brier, cross_fitted_scaling, ece, fit_temperature, scale
from src.make_docs import build, render
from src.utils import load_config

METRICS = Path(load_config("configs/app.yaml")["paths"]["metrics_json"])


@pytest.mark.skipif(not METRICS.exists(), reason="no metrics.json")
def test_generated_docs_are_up_to_date():
    """Fails if README.md or docs/MODEL_CARD.md was edited by hand or metrics.json changed:
    rerun `python -m src.make_docs --config configs/app.yaml`."""
    for path, text in build(METRICS).items():
        assert Path(path).read_text() == text, f"{path} is out of date"


def test_render_refuses_unknown_placeholders():
    assert render("BA {{ba}}", {"ba": "0.40"}, "t") == "BA 0.40"
    with pytest.raises(SystemExit, match="typo"):
        render("BA {{typo}}", {"ba": "0.40"}, "t")


def test_calibration_measures():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert brier(probs, np.array([0, 1])) == 0                    # sure and right
    assert brier(probs, np.array([1, 0])) == 2                    # sure and wrong
    conf = np.array([0.8] * 10)
    assert np.isclose(ece(conf, np.array([1] * 8 + [0] * 2, float), 10), 0)    # 80% sure, 80% right
    assert np.isclose(ece(conf, np.array([1] * 4 + [0] * 6, float), 10), 0.4)  # 80% sure, 40% right
    p = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(scale(p, 1.0), p)                          # T = 1 changes nothing
    assert scale(p, 2.0).max() < p.max() < scale(p, 0.5).max()    # T > 1 softens, T < 1 sharpens


def test_temperature_scaling_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 3
    logits = rng.normal(size=(n, k)) * 1.5
    true_p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in true_p])
    overconfident = scale(true_p, 0.5)                            # probabilities made too sharp
    grid = np.geomspace(0.25, 4, 200)
    assert 1.8 < fit_temperature(overconfident, y, grid) < 2.2   # T = 2 undoes the sharpening
    scaled, temps = cross_fitted_scaling(overconfident, y, np.arange(n) % 5, grid)
    assert len(temps) == 5 and ece(scaled.max(1), (scaled.argmax(1) == y).astype(float), 10) < \
        ece(overconfident.max(1), (overconfident.argmax(1) == y).astype(float), 10)
