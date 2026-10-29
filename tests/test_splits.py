"""Checks on labels and splits.

Unit tests use small synthetic data. Integration tests check the real
data/processed/cases.csv against the raw SCIN metadata, and are skipped if
the data has not been downloaded and built yet.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_metadata import EFST_MISSING, assign_splits
from src.data import (IMAGE_PATH_COLUMNS, add_eda_columns, assign_class,
                      duplicate_groups, load_scin, parse_weighted_label)
from src.utils import load_config

CONFIG = load_config("configs/labels.yaml")
CONDITION_TO_CLASS = {n: c for c, names in CONFIG["classes"].items() for n in names}


# ---------- unit tests ----------

def test_assign_class_rules():
    assert assign_class({"Eczema": 0.7, "Psoriasis": 0.3}, CONDITION_TO_CLASS) == "dermatitis"
    # tie inside one class is kept
    assert assign_class({"Eczema": 0.4, "Allergic Contact Dermatitis": 0.4, "Tinea": 0.2},
                        CONDITION_TO_CLASS) == "dermatitis"
    # tie across two kept classes is dropped
    assert assign_class({"Eczema": 0.5, "Psoriasis": 0.5}, CONDITION_TO_CLASS) is None
    # tie with a dropped condition is dropped
    assert assign_class({"Urticaria": 0.5, "Drug Rash": 0.5}, CONDITION_TO_CLASS) is None
    # top condition not kept
    assert assign_class({"Acne": 0.8, "Eczema": 0.2}, CONDITION_TO_CLASS) is None
    assert assign_class({}, CONDITION_TO_CLASS) is None


def test_duplicate_groups_are_transitive():
    images = pd.DataFrame({
        "case_id": ["a", "b", "b", "c", "d"],
        "image_path": ["x", "x", "y", "y", "z"],
    })
    groups = duplicate_groups(images)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_assign_splits_keeps_groups_together():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "case_id": [f"c{i}" for i in range(n)],
        "group_id": [f"g{i // 2}" for i in range(n)],  # pairs of cases share a group
        "label": rng.choice(["a", "b", "c"], n),
        "efst_group": rng.choice(["FST I-II", "FST III-IV", None], n),
    })
    out = assign_splits(df, test_size=0.2, n_cv_folds=5, seed=0)
    assert out.groupby("group_id").split.nunique().max() == 1
    assert out.groupby("group_id").cv_fold.nunique().max() == 1
    assert abs((out.split == "test").mean() - 0.2) < 0.05
    assert set(out.loc[out.split == "cv", "cv_fold"]) == set(range(5))
    assert (out.loc[out.split == "test", "cv_fold"] == -1).all()


# ---------- integration tests on the built metadata ----------

PROCESSED = Path(CONFIG["paths"]["processed_csv"])
RAW_DIR = Path(CONFIG["paths"]["raw_dir"])
needs_data = pytest.mark.skipif(
    not (PROCESSED.exists() and (RAW_DIR / "dataset/scin_cases.csv").exists()),
    reason="run src.download_scin and src.build_metadata first")


@pytest.fixture(scope="module")
def cases():
    return pd.read_csv(PROCESSED, dtype={"case_id": str, "group_id": str})


@pytest.fixture(scope="module")
def raw():
    return add_eda_columns(load_scin(RAW_DIR)).set_index("case_id")


def _partition(cases):
    """test, cv0..cv4 as one partition label per case."""
    return np.where(cases.split == "test", "test", "cv" + cases.cv_fold.astype(str))


@needs_data
def test_no_group_in_two_partitions(cases):
    part = cases.assign(part=_partition(cases))
    assert part.groupby("group_id").part.nunique().max() == 1


@needs_data
def test_no_image_in_two_partitions(cases):
    part = cases.assign(part=_partition(cases))
    images = part.melt(id_vars="part", value_vars=IMAGE_PATH_COLUMNS).dropna(subset=["value"])
    assert images.groupby("value").part.nunique().max() == 1


@needs_data
def test_missing_bucket_images_not_referenced(cases):
    referenced = set(cases[IMAGE_PATH_COLUMNS].stack())
    assert not referenced & set(CONFIG["missing_images"])


@needs_data
def test_every_case_maps_to_exactly_one_class(cases, raw):
    assert cases.case_id.is_unique
    assert set(cases.label) <= set(CONFIG["classes"])
    recomputed = raw.loc[cases.case_id, "weighted_skin_condition_label"].map(
        lambda s: assign_class(parse_weighted_label(s), CONDITION_TO_CLASS))
    assert (recomputed.values == cases.label.values).all()
    assert raw.loc[cases.case_id, "gradable"].all()


@needs_data
def test_split_counts_match_metadata(cases, raw):
    """Counts in reports/split_counts.csv equal counts recomputed from raw SCIN."""
    tone = raw.loc[cases.case_id, "efst_group"].fillna(EFST_MISSING).values
    recomputed = (cases.assign(efst_group=tone)
                  .groupby(["split", "label", "efst_group"]).size().rename("cases"))
    reported = (pd.read_csv(CONFIG["paths"]["split_counts_csv"])
                .set_index(["split", "label", "efst_group"]).cases)
    pd.testing.assert_series_equal(recomputed.sort_index(), reported.sort_index(),
                                   check_dtype=False)
    # Monk group stored in the CSV matches the raw Monk US label
    monk = raw.loc[cases.case_id, "monk_us_group"].values
    assert (pd.Series(monk).fillna("na").values == cases.monk_us_group.fillna("na").values).all()
