"""Phase 5 checks: configs differ only in image processing, augmentation is training-only,
preprocessing is deterministic and does what it says, and the paired comparison is sound."""
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("cv2")

from src.compare_settings import compare_subset, select_setting  # noqa: E402
from src.evaluate import EFST_ORDER  # noqa: E402
from src.transforms import CLAHE, ShadesOfGray, eval_transform, train_transform  # noqa: E402
from src.utils import load_config, merge  # noqa: E402

SETTINGS = load_config("configs/phase5/compare.yaml")["settings"]


def _random_ops(transform):
    return [t for t in transform.transforms if type(t).__name__.startswith("Random")]


def test_merge_overrides_nested_values_only():
    base = {"a": 1, "images": {"augmentation": "none", "input_size": 224}}
    assert merge(base, {"images": {"augmentation": "standard"}}) == \
        {"a": 1, "images": {"augmentation": "standard", "input_size": 224}}


def test_settings_differ_only_in_image_processing():
    configs = [load_config(p) for p in SETTINGS]
    assert len({c["experiment"] for c in configs}) == len(configs)
    strip = [{k: v for k, v in c.items() if k not in ("experiment", "images")} for c in configs]
    assert all(s == strip[0] for s in strip)
    images = [{k: v for k, v in c["images"].items() if k not in ("augmentation", "preprocessing")}
              for c in configs]
    assert all(i == images[0] for i in images)
    assert configs[0]["images"]["augmentation"] == "none"
    assert configs[0]["images"]["preprocessing"] == "none"
    assert configs[0]["train"]["epochs"] is None          # budget re-picked for every setting


@pytest.mark.parametrize("path", SETTINGS)
def test_augmentation_only_in_training(path):
    images = load_config(path)["images"]
    assert not _random_ops(eval_transform(images))
    assert bool(_random_ops(train_transform(images))) == (images["augmentation"] != "none")


@pytest.mark.parametrize("path", SETTINGS)
def test_preprocessing_applied_in_train_and_eval(path):
    images = load_config(path)["images"]
    expected = {"none": None, "shades_of_gray": ShadesOfGray, "clahe": CLAHE}[images["preprocessing"]]
    for transform in (train_transform(images), eval_transform(images)):
        first = transform.transforms[0]
        assert isinstance(first, expected) if expected else \
            not isinstance(first, (ShadesOfGray, CLAHE))


def test_preprocessing_is_deterministic():
    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((60, 80, 3)) * 255).astype(np.uint8))
    images = load_config(SETTINGS[-1])["images"]   # clahe
    a, b = eval_transform(images)(image), eval_transform(images)(image)
    assert (a == b).all()


def test_shades_of_gray_removes_a_colour_cast():
    rng = np.random.default_rng(0)
    grey = rng.integers(60, 200, (50, 50, 1)).repeat(3, axis=2).astype(np.float64)
    cast = np.clip(grey * [1.3, 1.0, 0.7], 0, 255).astype(np.uint8)   # warm (orange) light
    out = np.asarray(ShadesOfGray(6)(Image.fromarray(cast)), dtype=float)
    spread_before = np.ptp(cast.reshape(-1, 3).mean(axis=0))
    spread_after = np.ptp(out.reshape(-1, 3).mean(axis=0))
    assert spread_after < 0.2 * spread_before
    neutral = Image.fromarray(grey.astype(np.uint8))
    assert np.abs(np.asarray(ShadesOfGray(6)(neutral), float) - grey).max() <= 1  # already neutral


def test_clahe_raises_contrast_without_shifting_colour():
    rng = np.random.default_rng(0)
    flat = (120 + rng.integers(0, 15, (64, 64, 1))).repeat(3, axis=2).astype(np.uint8)
    flat[..., 0] += 20   # a reddish, low-contrast image
    out = np.asarray(CLAHE(2.0, 8)(Image.fromarray(flat)), dtype=float)
    assert out.std() > flat.std()
    assert (out[..., 0] > out[..., 2]).mean() > 0.95   # still reddish


def _probs(pred, classes, confidence=0.8):
    p = np.full((len(pred), len(classes)), (1 - confidence) / (len(classes) - 1))
    p[np.arange(len(pred)), [classes.index(c) for c in pred]] = confidence
    return p


def test_paired_comparison():
    rng = np.random.default_rng(0)
    classes, n = ["a", "b", "psoriasis"], 300
    y = rng.choice(classes, n)
    noisy = np.where(rng.random(n) < 0.5, y, rng.choice(classes, n))
    base = np.stack([_probs(noisy, classes)] * 2)          # 2 identical seeds
    better = np.stack([_probs(np.where(rng.random(n) < 0.5, y, noisy), classes)] * 2)
    ev = {"n_bootstrap": 200, "ci_level": 0.95}
    absolute, paired = compare_subset(y, {"base": base, "same": base.copy(), "better": better},
                                      classes, ["psoriasis"], ev, 5, rng)
    same = paired["same"]["balanced_accuracy"]
    assert same["difference"] == 0 and same["ci_low"] == same["ci_high"] == 0
    assert not same["outside_noise"]
    assert paired["better"]["balanced_accuracy"]["outside_noise"]
    assert paired["better"]["balanced_accuracy"]["difference"] > 0
    assert absolute["base"]["psoriasis_recall"]["n_cases"] == int((y == "psoriasis").sum())


def _paired(gain_low, group_high):
    gain = {"balanced_accuracy": {"difference": 0.05, "ci_low": gain_low, "ci_high": 0.1}}
    group = {"balanced_accuracy": {"difference": 0.0, "ci_low": -0.1, "ci_high": group_high}}
    return gain, group


def test_selection_rule():
    names = ["none", "aug"]
    results = {"overall": {"aug": {"balanced_accuracy": {"mean": 0.45}}}}
    for gain_low, group_high, expected in [(0.01, 0.05, "aug"),    # real gain, no harm
                                           (-0.01, 0.05, "none"),  # gain within noise
                                           (0.01, -0.01, "none")]: # a group made worse
        gain, group = _paired(gain_low, group_high)
        paired = {"overall": {"aug": gain}, **{g: {"aug": group} for g in EFST_ORDER}}
        assert select_setting(names, results, paired, "balanced_accuracy")["selected"] == expected

