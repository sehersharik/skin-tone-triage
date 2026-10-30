# Skin Condition Triage Across Skin Tones

Classifies common skin conditions from photos with a fine-tuned CNN, explains predictions with Grad-CAM,
and reports how performance varies across skin tones. Educational project, not a medical device.

**Status:** Phase 3 (baseline) in progress. Results, method and limitations will be written up in Phase 8.

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
