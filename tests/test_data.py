import numpy as np
import pandas as pd

from src.data import (aggregate_efst, fst_to_int, image_table,
                      parse_weighted_label, primary_condition)


def test_parse_weighted_label():
    assert parse_weighted_label("{'Eczema': 0.69, 'Psoriasis': 0.31}") == {
        "Eczema": 0.69, "Psoriasis": 0.31}
    assert parse_weighted_label("{}") == {}
    assert parse_weighted_label(np.nan) == {}


def test_primary_condition_clear_and_tied():
    assert primary_condition({"Eczema": 0.69, "Psoriasis": 0.31}) == ("Eczema", 0.69, False)
    name, weight, tied = primary_condition({"A": 0.41, "B": 0.41, "C": 0.18})
    assert tied and weight == 0.41 and name in {"A", "B"}
    name, weight, tied = primary_condition({})
    assert name is None and np.isnan(weight) and not tied


def test_fst_parsing_and_aggregation():
    assert fst_to_int("FST4") == 4
    assert np.isnan(fst_to_int("NONE_IDENTIFIED"))
    assert aggregate_efst(["FST2", "FST3", np.nan]) == 3  # median 2.5 rounds up
    assert aggregate_efst(["FST2", "FST2", "FST5"]) == 2
    assert np.isnan(aggregate_efst([np.nan, np.nan]))


def test_image_table_one_row_per_image():
    df = pd.DataFrame({
        "case_id": ["a", "b"],
        "image_1_path": ["a1.png", "b1.png"], "image_1_shot_type": ["CLOSE_UP"] * 2,
        "image_2_path": ["a2.png", np.nan], "image_2_shot_type": ["AT_AN_ANGLE", np.nan],
        "image_3_path": [np.nan, np.nan], "image_3_shot_type": [np.nan, np.nan],
    })
    images = image_table(df)
    assert len(images) == 3
    assert images.groupby("case_id").size().to_dict() == {"a": 2, "b": 1}
