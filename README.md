# Skin Condition Triage Across Skin Tones

<!-- Generated from docs/templates/README.md by `python -m src.make_docs --config configs/app.yaml`.
     Every number comes from reports/metrics.json. Edit the template, not this file. -->

A fine-tuned CNN that classifies 6 common skin conditions from consumer phone photos, explains its
predictions with Grad-CAM, and reports how its performance varies across skin tones. Educational project, not a
medical device.

## The honest conclusion

- **Better than chance, far from clinically useful.** On the held-out test set, balanced accuracy is
  **0.45 (0.37–0.54)** (95% CI); in cross-validation it was 0.38 (0.34–0.42). Guessing at random scores about 0.17.
  The test and CV numbers agree (within the noise).
- **The apparent skin-tone gap is mostly about which conditions each group has.** In CV the model looked worse on
  light skin (FST I-II) than on FST III-IV, but most of that gap disappears once the condition is held fixed, and
  what remains is borderline. It is driven mainly by psoriasis and tinea on light skin, and it does not clearly
  reappear on the test set.
- **Darker skin is essentially untested.** SCIN has few photos of darker skin: the test set has
  15 FST V-VI cases (no Folliculitis), where balanced accuracy was 0.20 (0.07–0.54), and
  3 Monk 7-10 cases. Nothing can be concluded either way, and that is itself the main limitation.
- **Image processing didn't beat the baseline on the pre-registered primary metric.** Standard augmentation
  improved macro F1, AUC and light-skin psoriasis recall, but missed the bar on balanced accuracy and needs
  15 epochs instead of 5. Colour constancy and CLAHE made no clear difference.
- **When the model is right, it mostly looks at the skin; when it is wrong, clothing, backgrounds and blacked-out
  areas are often part of why** (Grad-CAM).

## Problem

People increasingly photograph skin problems with their phones. A model that suggests a likely condition could
help triage, but only if it works equally well across skin tones, and dermatology datasets are known to
under-represent darker skin. This project asks: how well does a simple, standard CNN do on consumer photos, and
does its performance differ by skin tone? The emphasis is on careful, honest evaluation rather than a high score.

## Data

[SCIN](https://github.com/google-research-datasets/scin) (Skin Condition Image Network, Google Research and
Stanford Medicine): crowdsourced phone photos from people in the US, 1-3 photos per contribution ("case"), with
condition labels from dermatologists and skin-tone labels (dermatologist-estimated Fitzpatrick type, eFST, and
Monk skin tone from lay graders).

- **Labels:** 6 conditions with enough cases: Dermatitis (eczema), Folliculitis, Insect bite, Psoriasis, Tinea (ringworm), Urticaria (hives). Each case takes the dermatologists'
  top-weighted condition; cases whose top conditions tie across classes are dropped.
- **Splits at the case level** (a person's photos never appear in two splits; cases that share a photo are kept
  together): 80% for 5-fold cross-validation, 20% locked away as the test set, stratified by condition x skin
  tone. 928 CV cases (2054 photos) and 233 test cases (484 photos).

| Condition | CV cases | Test cases |
|---|---|---|
| Dermatitis (eczema) | 507 | 128 |
| Folliculitis | 75 | 17 |
| Insect bite | 107 | 27 |
| Psoriasis | 51 | 14 |
| Tinea (ringworm) | 49 | 11 |
| Urticaria (hives) | 139 | 36 |

## Method

- **Model:** EfficientNet-B0 pretrained on ImageNet (torchvision), new 6-class head, class-weighted cross-entropy,
  AdamW, cosine schedule, 224x224 centre crops. No augmentation in the frozen method.
- **Every choice on CV only:** the epoch budget was picked on one CV fold; settings were compared with 5-fold CV
  and 3 seeds; the final setting was chosen with a rule written down before the results existed.
- **Test set scored once**, for the frozen method trained on all CV cases. A guard in `src/test_eval.py` refuses
  to score it again.
- **Uncertainty everywhere:** 95% bootstrap confidence intervals (resampling cases) for every metric, and a gap is
  called real only if its interval excludes zero. Metrics are case level: photo probabilities are averaged per case.

## Results

### Overall

| Metric | CV, seed 42 | CV, 3 seeds (mean ± SD) | Test (233 cases) | Test minus CV |
|---|---|---|---|---|
| Balanced accuracy | 0.38 (0.34–0.42) | 0.39 ± 0.03 (0.36–0.42) | 0.45 (0.37–0.54) | +7.3 pts (-2.0 to +17.2) |
| Macro F1 | 0.36 (0.32–0.39) | 0.37 ± 0.03 (0.34–0.40) | 0.40 (0.33–0.48) | +4.9 pts (-3.5 to +13.6) |
| Macro AUC | 0.75 (0.73–0.78) | 0.76 ± 0.02 (0.74–0.78) | 0.79 (0.74–0.83) | +3.5 pts (-1.9 to +8.8) |

\* = outside the noise (95% CI excludes 0). Figures: [CV confusion matrix](docs/figures/baseline_confusion_matrix.png),
[test confusion matrix](docs/figures/test_confusion_matrix.png).

### Per condition (test set)

| Condition | Test cases | Recognised | Recall |
|---|---|---|---|
| Dermatitis (eczema) | 128 | 61 | 48% |
| Folliculitis | 17 | 7 | 41% |
| Insect bite | 27 | 14 | 52% |
| Psoriasis | 14 | 4 | 29% |
| Tinea (ringworm) | 11 | 4 | 36% |
| Urticaria (hives) | 36 | 24 | 67% |

Dermatitis, the most common class, is often confused with urticaria, insect bites and tinea.

### Per skin tone

Fitzpatrick (eFST) groups, the primary analysis:

| Group | CV cases | CV balanced accuracy (3-seed mean) | Test cases | Test balanced accuracy | Test AUC |
|---|---|---|---|---|---|
| FST I-II | 402 | 0.34 (0.30–0.39) | 102 | 0.45 (0.32–0.59) | 0.82 (0.75–0.88) |
| FST III-IV | 425 | 0.45 (0.39–0.49) | 106 | 0.49 (0.35–0.62) | 0.77 (0.68–0.84) |
| FST V-VI | 61 | 0.44 (0.30–0.64) | 15 | 0.20 (0.07–0.54) (weak evidence) | 0.78 (0.55–0.96) |

Monk (US) groups, a secondary check:

| Group | CV cases | CV balanced accuracy (3-seed mean) | Test cases | Test balanced accuracy | Test AUC |
|---|---|---|---|---|---|
| Monk 1-3 | 661 | 0.37 (0.34–0.41) | 170 | 0.48 (0.38–0.60) | 0.81 (0.75–0.86) |
| Monk 4-6 | 232 | 0.44 (0.38–0.51) | 59 | 0.38 (0.23–0.54) | 0.71 (0.58–0.83) |
| Monk 7-10 | 32 | 0.53 (0.31–0.72) | 3 | 0.67 (0.00–1.00) (weak evidence) | can't be computed |

Test results for small groups, or groups missing a condition, are marked as weak evidence. Figures:
[CV, 3 seeds](docs/figures/phase4_by_skin_tone_seeds.png), [test](docs/figures/test_by_skin_tone.png).

### Is the skin-tone gap real? (Phase 4)

In CV, FST III-IV beat FST I-II on balanced accuracy by +10.4 pts (+3.9 to +16.7) *. Balanced accuracy gives each condition
equal weight, so a few small conditions can drive it. Holding the condition fixed:

- Dermatitis only (enough cases in every group): +2.8 pts (-4.8 to +10.7)
- Accuracy, crude: +4.7 pts (-1.0 to +10.6); with every group given the same condition mix: +5.5 pts (+0.0 to +11.2)
- Logistic regression `correct ~ tone + condition`: odds ratio 1.26 (0.98–1.60) for FST III-IV vs I-II; adding photo
  brightness, contrast, sharpness and resolution: 1.29 (1.01–1.65)

The gap comes mainly from two small classes on light skin: psoriasis recall was 2% of 18 cases in
FST I-II vs 30% of 28 in III-IV, and tinea 15% of 20 vs 30% of 22. On the test set the
FST III-IV minus I-II gap is +4.4 pts (-15.1 to +22.4). Photo quality doesn't explain it. Figure:
[gap checks](docs/figures/phase4_gap_checks_efst.png).

### Image processing (Phase 5)

Four settings, 3 seeds x 5 folds each, the epoch budget re-picked per setting. Changes are paired
against the baseline on the same cases (percentage points, 95% CI; \* = outside the noise).

| Setting | Epochs (picked on fold 1) | Balanced accuracy (seed mean) | Δ balanced accuracy | Δ macro F1 | Δ macro AUC | Δ psoriasis recall, FST I-II |
|---|---|---|---|---|---|---|
| Baseline (none) | 5 | 0.39 (0.36–0.42) | — | — | — | — |
| Standard augmentation | 15 | 0.42 (0.39–0.45) | +2.3 pts (-0.6 to +5.0) | +5.0 pts (+2.2 to +7.8) * | +3.9 pts (+2.4 to +5.4) * | +20.4 pts (+7.4 to +35.4) * |
| Shades of Gray | 5 | 0.41 (0.38–0.44) | +1.6 pts (-0.3 to +3.5) | +1.6 pts (-0.1 to +3.5) | -0.5 pts (-1.5 to +0.5) | +0.0 pts (+0.0 to +0.0) |
| CLAHE | 5 | 0.40 (0.37–0.44) | +1.1 pts (-0.8 to +3.0) | +1.7 pts (-0.1 to +3.5) | +0.3 pts (-0.5 to +1.1) | +3.7 pts (+0.0 to +9.3) |

The pre-registered rule (a setting must improve overall balanced accuracy beyond the noise without making any
FST group worse) keeps **Baseline (none)**. Augmentation is a documented secondary finding, not adopted:
better macro F1, AUC and light-skin psoriasis recall, but it missed the primary metric, needs 15
epochs instead of 5, and tinea recall dropped (borderline). With 60 comparisons, an isolated
"outside the noise" result is weak evidence. Figure: [paired changes](docs/figures/phase5_paired_vs_baseline.png).

### Where the model looks (Grad-CAM, Phase 6)

Heatmaps from the last convolutional layer on all 2054 CV photos. On average only
19% (19%–20%) of the heatmap falls in the outer 10% border, against 35% if the model
looked everywhere equally: it mostly looks at the middle, where people put the lesion. Correctly predicted cases
put about 14% of the heatmap on the border; wrongly predicted cases 23%–25%, in every
FST group. In example grids, correct predictions sit on the plaque, bite or patch on light, medium and dark skin
alike, while wrong ones often land on clothing, background objects or blacked-out parts of the photo. No rulers
appeared in the cases inspected. The example grids show SCIN photos and are kept out of the repo; the
[mean heatmap per group](docs/figures/phase6_mean_gradcam.png) has no photos.

### How trustworthy are the probabilities?

On CV out-of-fold predictions, the expected calibration error (ECE) is 0.040 (0.036–0.082); its interval is lopsided
because resampling itself adds noise, which ECE counts as miscalibration. On average the model is +3.1 pts (+0.1 to +6.1)
overconfident, and in the middle of the range its probabilities match how often it is right.
**Very high probabilities are not reliable:** the 53 CV cases with a top probability of 80% or more
averaged 88% confidence but were right only 47% of the time. Temperature scaling
(fitted per CV fold, T = 1.02–1.16) didn't help (ECE change +0.005 (-0.021 to +0.024)). Because of the class-weighted
loss, the probabilities behave as if all conditions were equally common: dermatitis gets 32% of the
probability on average but is 55% of the cases. Figure: [reliability](docs/figures/calibration_cv.png).

## Limitations

- **Darker skin is barely represented** (61 CV and 15 test cases for FST V-VI), so
  performance there is unknown, not shown to be fine.
- **Only 6 conditions.** For anything else, including healthy skin, the model still picks one of them.
- **Modest accuracy and small groups:** the test set has 233 cases, so its intervals are wide.
- **Labels are dermatologists' opinions from photos alone,** without examination or history, and many cases had
  no clear single answer and were dropped.
- **US consumer photos only.** Results may not transfer to clinical photos or other populations.
- **Grad-CAM is coarse** (a 7x7 map) and shows where the model looked, not why it was right.
- **Probabilities above about 80% are overconfident**, and they don't reflect how common each condition is.

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

| Step | Command |
|---|---|
| Download SCIN (CSVs + sample) | `python -m src.download_scin --config configs/data.yaml` |
| Labels and case-level splits | `python -m src.build_metadata --config configs/labels.yaml` |
| Images of the kept cases | `python -m src.download_scin --config configs/data.yaml --images used` |
| Resize and cache | `python -m src.prepare_images --config configs/baseline.yaml` |
| Baseline CV + evaluation | `python -m src.train --config configs/baseline.yaml` then `python -m src.evaluate --config configs/baseline.yaml` |
| Skin-tone analysis (3 seeds) | `python -m src.train --config configs/seeds.yaml`, `python -m src.image_factors --config configs/seeds.yaml`, `python -m src.tone_analysis --config configs/seeds.yaml` |
| Image-processing settings | `python -m src.train --config configs/phase5/<setting>.yaml` for each, then `python -m src.compare_settings --config configs/phase5/compare.yaml` |
| Final model + Grad-CAM | `python -m src.train --config configs/final.yaml`, `python -m src.train_final --config configs/final.yaml`, `python -m src.gradcam --config configs/final.yaml` |
| Test set (once) | `python -m src.test_eval --config configs/test.yaml` |
| Calibration, app rule, docs | `python -m src.calibration --config configs/app.yaml`, `python -m src.app_confidence --config configs/app.yaml`, `python -m src.make_docs --config configs/app.yaml` |

Training needs a GPU: `notebooks/kaggle_colab_runner.ipynb` clones the repo and runs the steps on a free Kaggle
GPU. **Download the results zip from the version's Output tab; don't use Kaggle's "Save to GitHub" sync for this
notebook**, which commits the executed notebook straight to `main`. Every run logs its config and metrics to
`runs/`; plain-English notes on each phase are in [docs/learning-notes.md](docs/learning-notes.md).

## Demo app

```bash
streamlit run app/streamlit_app.py
```

Upload a photo to see the top guess, the probabilities for all 6 conditions, a Grad-CAM overlay and a
reliability note that is the same for everyone (the app doesn't guess your skin tone). Below 50%
top probability it says "uncertain"; in CV that is 58% of cases, and the top guess was right
58% of the time above the threshold vs 38% below. Photos are processed in
memory and not saved. The weights (`models/final_baseline.pt`) are not in git; see
[docs/DEPLOY.md](docs/DEPLOY.md) and the [model card](docs/MODEL_CARD.md).

## What I'd do next

- **External validation on DermaCon-IN** (clinical photos from India),
  which would test both a different population and much more darker skin.
- **More darker-skin data** for training and, above all, for testing, so per-tone results stop being weak
  evidence. Sources such as Fitzpatrick17k or DDI could add cases, with care over label differences.
- **Take augmentation further** with a pre-registered confirmation run, since it improved several secondary
  metrics.
- **Fix the high-confidence end** (for example, cap the displayed probability or recalibrate with class priors)
  and test whether lesion-focused crops reduce reliance on backgrounds.

## Data and license

This project uses the **SCIN dataset** by Google Research and Stanford Medicine,
<https://github.com/google-research-datasets/scin>, under the
[SCIN Data Use License](https://github.com/google-research-datasets/scin/blob/main/LICENSE). The data is provided
as-is, without warranty. No SCIN images are stored in this repository, and the license forbids any attempt to
re-identify contributors. This project is not endorsed by the dataset's creators.

> Ward A, Li J, Wang J, et al. *Creating an Empirical Dermatology Dataset Through Crowdsourcing With Web Search Advertisements.*
> JAMA Netw Open. 2024;7(11):e2446615. doi:10.1001/jamanetworkopen.2024.46615
