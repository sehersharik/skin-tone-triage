# Skin Condition Triage Across Skin Tones

Classifies common skin conditions from photos with a fine-tuned CNN, explains predictions with Grad-CAM,
and reports how performance varies across skin tones. Educational project, not a medical device.

**Status:** Phases 1-5 done; Phase 6 (Grad-CAM) in progress. The full results, method and limitations will be
written up in Phase 8. Every number lives in `reports/metrics.json`; plain-English notes per phase are in
[`docs/learning-notes.md`](docs/learning-notes.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get the data

SCIN is downloaded from its public Google Cloud Storage bucket using the method in the official
[`scin_demo.ipynb`](https://github.com/google-research-datasets/scin/blob/main/scin_demo.ipynb). No Google login is needed.

```bash
# CSVs + a 200-image random sample (enough for the EDA on a laptop)
python -m src.download_scin --config configs/data.yaml

# After building labels and splits (below): only the ~2.5k images of the kept cases (~2.9 GB)
python -m src.download_scin --config configs/data.yaml --images used
```

Each run logs its config and a summary to `runs/<timestamp>_download_scin/`.

## Build labels and splits

```bash
python -m src.build_metadata --config configs/labels.yaml
```

Writes `data/processed/cases.csv` (one row per case: class, skin-tone groups, split, CV fold) and
`reports/split_counts.csv`.

## Train and evaluate the baseline (GPU; see `notebooks/kaggle_colab_runner.ipynb`)

```bash
python -m src.prepare_images --config configs/baseline.yaml   # resize once, cache
python -m src.train --config configs/baseline.yaml            # 5-fold CV, out-of-fold predictions
python -m src.evaluate --config configs/baseline.yaml         # metrics + CIs -> reports/metrics.json
```

The test set is not used by any of these steps.

Each finished fold is saved to `reports/predictions/baseline_folds/` (or `--fold-outputs-dir`); rerunning
`src.train` after an interruption skips the saved folds and gives the same results.

## Skin-tone analysis (Phase 4) and image-processing experiments (Phase 5)

```bash
python -m src.train --config configs/seeds.yaml                  # baseline, 3 seeds, same folds
python -m src.image_factors --config configs/seeds.yaml          # brightness, contrast, sharpness, resolution
python -m src.tone_analysis --config configs/seeds.yaml          # -> tone_analysis_cv in reports/metrics.json

for s in none standard_aug shades_of_gray clahe; do
  python -m src.train --config configs/phase5/$s.yaml            # epoch budget re-picked per setting
done
python -m src.compare_settings --config configs/phase5/compare.yaml   # -> phase5_cv
```

## Demo app (Streamlit, CPU)

```bash
python -m src.app_confidence --config configs/app.yaml   # CV behaviour of the "uncertain" rule -> metrics.json
streamlit run app/streamlit_app.py
```

Upload a photo to see the predicted condition, the probabilities for all 6 classes, a Grad-CAM overlay and a
reliability note. Every number on the page is read from `reports/metrics.json`. Photos are processed in memory and
not saved. The app needs the model weights at `models/final_baseline.pt`; they are not in git (train them with
`src.train_final`, see `configs/final.yaml`). Without them the app explains that it can't predict.

## Running on Kaggle

`notebooks/kaggle_colab_runner.ipynb` clones this repo and runs the scripts for the phase set in its first cell.
**Download the results zip from the version's Output tab and unzip it in the repo root. Do not use Kaggle's
"Save to GitHub" sync for this notebook:** it commits the executed notebook (with all outputs) straight to
`main`, which conflicts with local commits.

## Findings so far (CV pool only; the test set has not been used)

Numbers and confidence intervals for everything below are in `reports/metrics.json`.

- **Frozen method: the baseline** (EfficientNet-B0, no augmentation, no preprocessing, 5 epochs). It was kept by a
  selection rule written down before the Phase 5 results existed: a setting had to improve overall balanced
  accuracy beyond the noise without making any skin-tone group worse. None did.
- **Secondary finding: standard augmentation.** Compared with the baseline on the same cases, it improved macro F1,
  macro AUC and psoriasis recall on light skin (FST I-II) beyond the noise, but it missed the pre-registered
  primary metric (its overall balanced-accuracy gain was within the noise). It costs about 3x the training time,
  and tinea recall dropped (borderline). It is documented here, not adopted.
- **Colour constancy (Shades of Gray) and CLAHE** made no clear difference.
- **Skin tone (Phase 4):** the baseline's lower balanced accuracy on FST I-II than on FST III-IV mostly disappears
  once the condition is held fixed; what remains is borderline and comes mainly from psoriasis and tinea in
  FST I-II. Photo quality doesn't explain it. The darkest groups are too small to draw conclusions.

## Explore

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
pytest -q
```

## Data and license

This project uses the **SCIN (Skin Condition Image Network) dataset** by Google Research and Stanford Medicine,
<https://github.com/google-research-datasets/scin>, released under the
[SCIN Data Use License](https://github.com/google-research-datasets/scin/blob/main/LICENSE).
The data is provided as-is, without warranty. No SCIN images are stored in this repository.
The license forbids any attempt to re-identify contributors.

> Ward A, Li J, Wang J, et al. *Creating an Empirical Dermatology Dataset Through Crowdsourcing With Web Search Advertisements.*
> JAMA Netw Open. 2024;7(11):e2446615. doi:10.1001/jamanetworkopen.2024.46615
