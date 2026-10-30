"""Training-side safety checks: no test cases loaded, no augmentation at eval time."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.transforms import eval_transform, train_transform  # noqa: E402
from src.utils import load_config  # noqa: E402

CONFIG = load_config("configs/baseline.yaml")


def _random_ops(transform):
    return [t for t in transform.transforms if type(t).__name__.startswith("Random")]


def test_eval_transform_has_no_augmentation():
    assert not _random_ops(eval_transform(CONFIG["images"]))


def test_baseline_train_transform_has_no_augmentation():
    assert CONFIG["images"]["augmentation"] == "none"
    assert not _random_ops(train_transform(CONFIG["images"]))


@pytest.mark.skipif(not Path(CONFIG["paths"]["cases_csv"]).exists(), reason="build metadata first")
def test_training_loads_cv_pool_only():
    import pandas as pd

    from src.train import load_cv_pool
    cv = load_cv_pool(CONFIG["paths"])
    all_cases = pd.read_csv(CONFIG["paths"]["cases_csv"], dtype={"case_id": str})
    test_ids = set(all_cases.loc[all_cases.split == "test", "case_id"])
    assert (cv.split == "cv").all()
    assert not test_ids & set(cv.case_id)


def test_class_weights_balance_classes():
    import numpy as np

    from src.train import class_weights
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    # 3 images of class 0 x weight + 1 image of class 1 x weight are equal
    assert torch.isclose(3 * w[0], 1 * w[1])


def test_fold_store_resumes_and_rejects_other_configs(tmp_path):
    import pandas as pd

    from src.train import FoldStore
    store = FoldStore(tmp_path, CONFIG)
    assert store.load("cv_fold_0_5ep") == (None, None)
    preds = pd.DataFrame({"case_id": ["007", "012"], "prob_a": [0.2, 0.9]})
    store.save("cv_fold_0_5ep", {"seconds": 12.5}, preds)

    info, loaded = FoldStore(tmp_path, CONFIG).load("cv_fold_0_5ep")  # a "rerun"
    assert info == {"seconds": 12.5}
    pd.testing.assert_frame_equal(loaded, preds)  # case ids keep their leading zeros

    other = {**CONFIG, "seed": CONFIG["seed"] + 1}
    with pytest.raises(SystemExit):
        FoldStore(tmp_path, other)
