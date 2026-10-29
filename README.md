# Skin Condition Triage Across Skin Tones

Classifies common skin conditions from photos with a fine-tuned CNN, explains predictions with Grad-CAM,
and reports how performance varies across skin tones. Educational project, not a medical device.

**Status:** Phase 2 (labels and splits) done. Results, method and limitations will be written up in Phase 8.

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

# CSVs + all ~10.4k images (~12.6 GB), e.g. on Kaggle/Colab
python -m src.download_scin --config configs/data.yaml --images all
```

Each run logs its config and a summary to `runs/<timestamp>_download_scin/`.

## Build labels and splits

```bash
python -m src.build_metadata --config configs/labels.yaml
```

Writes `data/processed/cases.csv` (one row per case: class, skin-tone groups, split, CV fold) and
`reports/split_counts.csv`.

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
