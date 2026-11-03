"""Phase 6 checks: Grad-CAM points at what drives the score, the border measure, case
selection uses out-of-fold correctness only, and the final config is the frozen method."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.gradcam import GradCAM, border_share, peak_in_border, select_cases  # noqa: E402
from src.utils import load_config  # noqa: E402


def toy_model():
    """features: a 1x1 conv that copies the input; head: global mean -> class 0 = +, class 1 = -."""
    features = torch.nn.Conv2d(1, 1, kernel_size=1, bias=False)
    features.weight.data.fill_(1.0)
    head = torch.nn.Linear(1, 2, bias=False)
    head.weight.data = torch.tensor([[1.0], [-1.0]])
    model = torch.nn.Sequential(features, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), head)
    return model, features


def test_gradcam_highlights_the_region_that_drives_the_class():
    model, layer = toy_model()
    x = torch.zeros(1, 1, 16, 16)
    x[0, 0, 2:6, 2:6] = 1.0                       # a bright patch, top left
    cam, probs = GradCAM(model, layer)(x, class_idx=0)
    assert cam.shape == (16, 16) and cam.max() == 1.0 and cam.min() >= 0
    assert cam[2:6, 2:6].mean() > 0.9 and cam[10:, 10:].max() < 0.1
    assert np.isclose(probs.sum(), 1)
    # class 1 is pushed DOWN by the patch -> nothing positive to show
    cam_other, _ = GradCAM(model, layer)(x, class_idx=1)
    assert cam_other.max() == 0


def test_model_with_gradcam_still_predicts_without_gradients():
    model, layer = toy_model()
    cam_fn = GradCAM(model, layer)
    x = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)        # e.g. the app's prediction step
    cam, cam_probs = cam_fn(x, int(probs.argmax()))
    assert np.allclose(probs[0].numpy(), cam_probs) and cam.shape == (8, 8)


def test_border_share():
    uniform = np.ones((100, 100))
    assert np.isclose(border_share(uniform, 0.1), 1 - 0.8 ** 2)
    centre = np.zeros((100, 100)); centre[40:60, 40:60] = 1
    assert border_share(centre, 0.1) == 0 and not peak_in_border(centre, 0.1)
    edge = np.zeros((100, 100)); edge[:5, :] = 1
    assert border_share(edge, 0.1) == 1 and peak_in_border(edge, 0.1)


def test_select_cases_follows_out_of_fold_correctness():
    rng = np.random.default_rng(0)
    n = 120
    oof = pd.DataFrame({"case_id": [f"{i:03d}" for i in range(n)],
                        "efst_group": rng.choice(["FST I-II", "FST III-IV"], n),
                        "label": rng.choice(["a", "psoriasis"], n),
                        "pred": rng.choice(["a", "psoriasis"], n)})
    oof["correct"] = oof.pred == oof.label
    gcfg = {"cases_per_class": 1, "cases_per_focus_class": 2, "focus_classes": ["psoriasis"]}
    picks, missing = select_cases(oof, ["a", "psoriasis"], gcfg, np.random.default_rng(1))
    assert (picks.correct == (picks.pred == picks.label)).all()
    counts = picks.groupby(["efst_group", "correct", "label"]).size()
    assert (counts.xs("psoriasis", level="label") == 2).all() and (counts.xs("a", level="label") == 1).all()
    assert {m["efst_group"] for m in missing} == {"FST V-VI"}      # no such cases in the toy data
    again, _ = select_cases(oof, ["a", "psoriasis"], gcfg, np.random.default_rng(1))
    assert picks.case_id.tolist() == again.case_id.tolist()        # seeded, reproducible


def test_final_config_is_the_frozen_baseline():
    final, base = load_config("configs/final.yaml"), load_config("configs/baseline.yaml")
    assert final["seed"] == 42 and final["train"]["epochs"] == 5
    for key in ("images", "model", "evaluation"):
        assert final[key] == base[key]
    assert {k: v for k, v in final["train"].items() if k != "epochs"} == \
        {k: v for k, v in base["train"].items() if k != "epochs"}
    assert final["images"]["augmentation"] == "none"
