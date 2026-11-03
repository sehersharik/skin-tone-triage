# Model card: skin-condition triage baseline (EfficientNet-B0 fine-tuned on SCIN)

<!-- Generated from docs/templates/MODEL_CARD.md by `python -m src.make_docs --config configs/app.yaml`.
     Every number comes from reports/metrics.json. Also used as the GitHub Release notes. -->

**Educational demo, not a medical device. Do not use it to make health decisions.**

## Model

- **File:** `final_baseline.pt` (PyTorch `state_dict`), SHA-256 `375c7ff779b3050c9d11e988cd82820fd37940718e6b31438c4844bca48b5f44`
- **Architecture:** EfficientNet-B0 (torchvision) with ImageNet weights, final layer replaced by a
  6-class layer: Dermatitis (eczema), Folliculitis, Insect bite, Psoriasis, Tinea (ringworm), Urticaria (hives).
- **Training:** fine-tuned on all 928 cross-validation cases (2054 photos) of the SCIN subset
  below, 5 epochs, seed 42, class-weighted cross-entropy, AdamW, cosine schedule,
  224x224 centre crops, no augmentation. Trained with PyTorch 2.11.0+cu130. Code and configs:
  <https://github.com/sehersharik/skin-tone-triage> (`configs/final.yaml`).
- **Input:** one RGB photo; resize the short side to 256 px, then 224 px, centre-crop 224x224, ImageNet
  normalisation. For a case with several photos, average the photos' probabilities.

## Intended use

- Teaching and portfolio demonstration of careful evaluation: per-skin-tone analysis with confidence intervals,
  a once-only test set, Grad-CAM.
- **Out of scope:** diagnosis, triage of real patients, any decision about health or treatment, use on
  conditions other than the 6 above, and estimating anyone's skin tone.

## Performance

Test set: 233 cases, scored once. 95% bootstrap CIs, resampling cases.

| Metric | CV, seed 42 | CV, 3 seeds (mean ± SD) | Test (233 cases) | Test minus CV |
|---|---|---|---|---|
| Balanced accuracy | 0.38 (0.34–0.42) | 0.39 ± 0.03 (0.36–0.42) | 0.45 (0.37–0.54) | +7.3 pts (-2.0 to +17.2) |
| Macro F1 | 0.36 (0.32–0.39) | 0.37 ± 0.03 (0.34–0.40) | 0.40 (0.33–0.48) | +4.9 pts (-3.5 to +13.6) |
| Macro AUC | 0.75 (0.73–0.78) | 0.76 ± 0.02 (0.74–0.78) | 0.79 (0.74–0.83) | +3.5 pts (-1.9 to +8.8) |

Per Fitzpatrick (eFST) group:

| Group | CV cases | CV balanced accuracy (3-seed mean) | Test cases | Test balanced accuracy | Test AUC |
|---|---|---|---|---|---|
| FST I-II | 402 | 0.34 (0.30–0.39) | 102 | 0.45 (0.32–0.59) | 0.82 (0.75–0.88) |
| FST III-IV | 425 | 0.45 (0.39–0.49) | 106 | 0.49 (0.35–0.62) | 0.77 (0.68–0.84) |
| FST V-VI | 61 | 0.44 (0.30–0.64) | 15 | 0.20 (0.07–0.54) (weak evidence) | 0.78 (0.55–0.96) |

## Known limitations

- **Darker skin is essentially untested.** The test set has 15 FST V-VI cases (none with
  Folliculitis), where balanced accuracy was 0.20 (0.07–0.54), and 3 Monk 7-10 cases.
- **Light-skin gap in cross-validation:** worse balanced accuracy on FST I-II than III-IV, mostly explained by
  the condition mix and by poor psoriasis and tinea recognition on light skin; not clearly seen on the test set.
- **Only 6 conditions.** Any other condition, or healthy skin, is still assigned one of them.
- **Overconfident at the top:** in cross-validation, top probabilities of 80% or more were right
  47% of the time. Probabilities behave as if all conditions were equally common (class-weighted
  training).
- **Shortcut risk:** Grad-CAM shows wrong predictions often rely on clothing, backgrounds or blacked-out areas.
- **Data:** US consumer phone photos labelled by dermatologists from photos alone. Results may not transfer to
  other populations or clinical photos.

## Training data, attribution and license

- **Dataset:** SCIN (Skin Condition Image Network) by Google Research and Stanford Medicine,
  <https://github.com/google-research-datasets/scin>. Citation: Ward A, Li J, Wang J, et al. *Creating an
  Empirical Dermatology Dataset Through Crowdsourcing With Web Search Advertisements.* JAMA Netw Open.
  2024;7(11):e2446615.
- **License:** the dataset is used under the
  [SCIN Data Use License](https://github.com/google-research-datasets/scin/blob/main/LICENSE). This model is
  shared on the basis that it may be Adapted Material under that license, and is subject to its terms.
- **What was changed:** a subset of SCIN was used (6 conditions; cases with a clear top label),
  photos were resized, and the model above was trained on them. No SCIN images are distributed with this model.
- **No re-identification:** you may not attempt to re-identify or re-link any person whose data was used to
  create SCIN. This is a condition of the SCIN license.
- **No warranty:** the SCIN data, and this model, are provided as-is, without warranties of any kind, including
  accuracy and fitness for a particular purpose.
- **No endorsement:** this model is not connected with, sponsored or endorsed by Google, Stanford Medicine or the
  SCIN creators.
