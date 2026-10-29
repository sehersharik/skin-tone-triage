"""Build the clean case-level metadata CSV with labels and splits.

Steps:
1. Map each case's weighted dermatologist label to one kept class (or drop it).
2. Group cases that share any image file, so duplicates never cross splits.
3. Hold out a test set (~20%) with StratifiedGroupKFold, stratified on
   class x eFST group and split by group.
4. Assign 5 grouped, stratified CV folds to the remaining cases.

Usage:
    python -m src.build_metadata --config configs/labels.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data import (IMAGE_PATH_COLUMNS, WEIGHTED_LABEL_COLUMN, add_eda_columns,
                      assign_class, duplicate_groups, image_table, load_scin,
                      parse_weighted_label, top_conditions)
from src.utils import load_config, save_metrics, set_seed, start_run

EFST_MISSING = "eFST missing"
OUTPUT_COLUMNS = [
    "case_id", "group_id", "label", "split", "cv_fold",
    "efst", "efst_group", "monk_us", "monk_us_group", "monk_india",
    "label_weight", "class_weight", "top_conditions", "top_tied",
    "n_images", *IMAGE_PATH_COLUMNS,
]


def label_cases(df, classes):
    """Add label, label_weight (top weight) and class_weight (summed weight of
    every condition in the case that maps to the assigned class)."""
    condition_to_class = {name: cls for cls, names in classes.items() for name in names}
    weights = df[WEIGHTED_LABEL_COLUMN].map(parse_weighted_label)
    top = weights.map(top_conditions)

    df = df.copy()
    df["label"] = weights.map(lambda w: assign_class(w, condition_to_class))
    df["label_weight"] = top.map(lambda t: t[1])
    df["top_conditions"] = top.map(lambda t: "; ".join(t[0]))
    df["top_tied"] = top.map(lambda t: len(t[0]) > 1)
    df["class_weight"] = [
        sum(v for k, v in w.items() if condition_to_class.get(k) == label) if label else np.nan
        for w, label in zip(weights, df["label"])
    ]
    return df


def drop_reasons(df, classes):
    """Count why cases were dropped (for the run log)."""
    kept_names = {name for names in classes.values() for name in names}
    labelled = df.n_conditions > 0
    top_in_kept = df.top_conditions.map(lambda s: any(n in kept_names for n in s.split("; ")))
    return {
        "all_cases": len(df),
        "no_condition_label_or_not_gradable": int((~(labelled & df.gradable)).sum()),
        "top_condition_not_in_kept_classes": int((labelled & df.gradable & ~top_in_kept).sum()),
        "tie_across_classes_or_with_dropped_condition": int(
            (labelled & df.gradable & top_in_kept & df.label.isna()).sum()),
        "kept": int(df.label.notna().sum()),
    }


def assign_splits(df, test_size, n_cv_folds, seed):
    """Return df with `split` (cv/test) and `cv_fold` (0..n-1, -1 for test)."""
    strata = df.label + " | " + df.efst_group.fillna(EFST_MISSING)

    # Test hold-out: one fold of a grouped stratified k-fold with k = 1 / test_size
    outer = StratifiedGroupKFold(n_splits=round(1 / test_size), shuffle=True, random_state=seed)
    _, test_idx = next(outer.split(df, strata, groups=df.group_id))
    df = df.copy()
    df["split"] = "cv"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    # CV folds on the remaining cases
    df["cv_fold"] = -1
    pool = df.index[df.split == "cv"]
    inner = StratifiedGroupKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(inner.split(pool, strata[pool], groups=df.group_id[pool])):
        df.loc[pool[val_idx], "cv_fold"] = fold
    return df


def split_counts(df):
    """Long table: cases and images per split x class x eFST group."""
    return (df.assign(efst_group=df.efst_group.fillna(EFST_MISSING))
              .groupby(["split", "label", "efst_group"])
              .agg(cases=("case_id", "size"), images=("n_images", "sum"))
              .reset_index())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    paths, split_cfg = config["paths"], config["split"]
    run_dir = start_run(paths["runs_dir"], "build_metadata", config)

    df = load_scin(paths["raw_dir"])
    # Remove references to image files that do not exist in the bucket
    for col in IMAGE_PATH_COLUMNS:
        df.loc[df[col].isin(config["missing_images"]), col] = np.nan
    df = add_eda_columns(df)

    # Groups are built over all cases, before filtering, to be conservative
    df["group_id"] = df.case_id.map(duplicate_groups(image_table(df)))

    df = label_cases(df, config["classes"])
    df["label"] = df.label.where(df.gradable)
    reasons = drop_reasons(df, config["classes"])

    kept = df[df.label.notna() & (df.n_images > 0)].reset_index(drop=True)
    kept = assign_splits(kept, split_cfg["test_size"], split_cfg["n_cv_folds"], config["seed"])

    out = Path(paths["processed_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    kept[OUTPUT_COLUMNS].sort_values(["split", "cv_fold", "case_id"]).to_csv(out, index=False)

    counts = split_counts(kept)
    Path(paths["split_counts_csv"]).parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(paths["split_counts_csv"], index=False)

    group_sizes = kept.group_id.value_counts()
    summary = {
        "case_filtering": reasons,
        "kept_cases": len(kept),
        "kept_images": int(kept.n_images.sum()),
        "groups": int(kept.group_id.nunique()),
        "groups_with_more_than_one_case": int((group_sizes > 1).sum()),
        "cases_per_split": kept.split.value_counts().to_dict(),
        "test_fraction_of_cases": round(float((kept.split == "test").mean()), 4),
        "cases_per_cv_fold": kept[kept.split == "cv"].cv_fold.value_counts().sort_index().to_dict(),
        "cases_per_class": kept.label.value_counts().to_dict(),
    }
    save_metrics(run_dir, summary)

    for split in ("cv", "test"):
        table = (counts[counts.split == split]
                 .pivot(index="label", columns="efst_group", values="cases")
                 .fillna(0).astype(int))
        table["total"] = table.sum(axis=1)
        table.loc["total"] = table.sum()
        print(f"\nCases per class x eFST group: {split}\n{table.to_string()}")
    print(f"\nWrote {out} ({len(kept)} cases). Run log: {run_dir}")


if __name__ == "__main__":
    main()
