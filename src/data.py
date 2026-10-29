"""Load and parse SCIN metadata.

One SCIN *case* is one contribution from one person, with 1-3 images.
SCIN publishes no separate contributor ID, so case_id is the unit we split on.

Definitions used here (EDA defaults, to be confirmed in Phase 2):
- primary condition: the condition with the highest weight in
  `weighted_skin_condition_label`. If two conditions share the top weight the
  case is flagged as tied (no single clear label).
- gradable: at least one dermatologist marked the case gradable for skin
  condition (DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT).
- eFST: median of the available dermatologist Fitzpatrick labels, rounded
  half up (e.g. FST2 + FST3 -> 3).
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_PATH_COLUMNS = ["image_1_path", "image_2_path", "image_3_path"]
# Column name as it appears in the released CSV (the schema doc omits "_on")
CONDITION_NAMES_COLUMN = "dermatologist_skin_condition_on_label_name"
WEIGHTED_LABEL_COLUMN = "weighted_skin_condition_label"
GRADABLE_YES = "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT"

FST_GROUPS = {1: "FST I-II", 2: "FST I-II", 3: "FST III-IV", 4: "FST III-IV",
              5: "FST V-VI", 6: "FST V-VI"}
MONK_GROUPS = {**dict.fromkeys((1, 2, 3), "Monk 1-3"),
               **dict.fromkeys((4, 5, 6), "Monk 4-6"),
               **dict.fromkeys((7, 8, 9, 10), "Monk 7-10")}


def load_scin(raw_dir):
    """Merge scin_cases.csv and scin_labels.csv on case_id (one row per case)."""
    dataset_dir = Path(raw_dir) / "dataset"
    cases = pd.read_csv(dataset_dir / "scin_cases.csv", dtype={"case_id": str})
    labels = pd.read_csv(dataset_dir / "scin_labels.csv", dtype={"case_id": str})
    return cases.merge(labels, on="case_id", how="left", validate="one_to_one")


def parse_weighted_label(value):
    """"{'Eczema': 0.69, ...}" -> dict. Missing or empty -> {}."""
    if not isinstance(value, str):
        return {}
    return ast.literal_eval(value)


def top_conditions(weights):
    """Return ([all conditions sharing the top weight], top weight)."""
    if not weights:
        return [], np.nan
    top_weight = max(weights.values())
    return [name for name, w in weights.items() if w == top_weight], top_weight


def primary_condition(weights):
    """Return (condition, weight, is_tied) for the top-weighted condition."""
    top, top_weight = top_conditions(weights)
    if not top:
        return None, np.nan, False
    return top[0], top_weight, len(top) > 1


def assign_class(weights, condition_to_class):
    """Map a case's weighted label to one kept class, or None to drop it.

    The case is kept only if every condition sharing the top weight maps to the
    same kept class (so an Eczema / Allergic Contact Dermatitis tie -> dermatitis).
    """
    top, _ = top_conditions(weights)
    classes = {condition_to_class.get(name) for name in top}
    if len(classes) == 1 and None not in classes:
        return classes.pop()
    return None


def duplicate_groups(images):
    """Give each case a group id so that cases sharing any image file share a group.

    `images` has one row per image (case_id, image_path). Uses union-find:
    every case starts in its own group, and cases are merged whenever they
    reference the same image path. The group id is the smallest case_id in it.
    """
    parent = {case: case for case in images.case_id.unique()}

    def find(case):
        while parent[case] != case:
            parent[case] = parent[parent[case]]
            case = parent[case]
        return case

    for _, cases in images.groupby("image_path").case_id:
        first, *rest = cases.unique()
        for other in rest:
            a, b = sorted([find(first), find(other)])
            parent[b] = a
    return {case: find(case) for case in parent}


def fst_to_int(value):
    """'FST3' -> 3. Anything else (NaN, NONE_IDENTIFIED) -> NaN."""
    if isinstance(value, str) and value.startswith("FST") and value[3:].isdigit():
        return int(value[3:])
    return np.nan


def aggregate_efst(labels):
    """Median of the available dermatologist FST labels, rounded half up."""
    values = [v for v in (fst_to_int(x) for x in labels) if not np.isnan(v)]
    if not values:
        return np.nan
    return int(np.floor(np.median(values) + 0.5))


def add_eda_columns(df):
    """Add the derived columns used in the EDA notebook."""
    df = df.copy()
    weights = df[WEIGHTED_LABEL_COLUMN].map(parse_weighted_label)
    top = weights.map(primary_condition)
    df["primary_condition"] = top.map(lambda t: t[0])
    df["primary_weight"] = top.map(lambda t: t[1])
    df["primary_tied"] = top.map(lambda t: t[2])
    df["n_conditions"] = weights.map(len)

    gradable_cols = df.filter(like="dermatologist_gradable_for_skin_condition_").columns
    df["gradable"] = (df[gradable_cols] == GRADABLE_YES).any(axis=1)
    df["n_images"] = df[IMAGE_PATH_COLUMNS].notna().sum(axis=1)

    efst_cols = df.filter(like="dermatologist_fitzpatrick_skin_type_label_").columns
    df["efst"] = df[efst_cols].apply(lambda row: aggregate_efst(row.tolist()), axis=1)
    df["efst_group"] = df["efst"].map(FST_GROUPS)
    df["monk_us"] = df["monk_skin_tone_label_us"]
    df["monk_india"] = df["monk_skin_tone_label_india"]
    df["monk_us_group"] = df["monk_us"].map(MONK_GROUPS)
    return df


def image_table(df):
    """One row per image: case_id, image_path, shot_type."""
    rows = []
    for i in (1, 2, 3):
        part = df[["case_id", f"image_{i}_path", f"image_{i}_shot_type"]].dropna(
            subset=[f"image_{i}_path"])
        part.columns = ["case_id", "image_path", "shot_type"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)
