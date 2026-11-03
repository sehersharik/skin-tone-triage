# Skin Condition Triage Across Skin Tones

<!-- Generated from docs/templates/README.md by `python -m src.make_docs --config configs/app.yaml`.
     Every number comes from reports/metrics.json. Edit the template, not this file. -->

A fine-tuned CNN that classifies {{n_classes}} common skin conditions from consumer phone photos, explains its
predictions with Grad-CAM, and reports how its performance varies across skin tones. Educational project, not a
medical device.

## The honest conclusion

- **Better than chance, far from clinically useful.** On the held-out test set, balanced accuracy is
  **{{test_ba}}** (95% CI); in cross-validation it was {{cv_ba}}. Guessing at random scores about {{chance}}.
  The test and CV numbers agree ({{test_cv_agree}}).
- **The apparent skin-tone gap is mostly about which conditions each group has.** In CV the model looked worse on
  light skin (FST I-II) than on FST III-IV, but most of that gap disappears once the condition is held fixed, and
  what remains is borderline. It is driven mainly by psoriasis and tinea on light skin, and it does not clearly
  reappear on the test set.
- **Darker skin is essentially untested.** SCIN has few photos of darker skin: the test set has
  {{fst56_test_n}} FST V-VI cases (no {{fst56_absent}}), where balanced accuracy was {{fst56_test_ba}}, and
  {{monk710_test_n}} Monk 7-10 cases. Nothing can be concluded either way, and that is itself the main limitation.
- **Image processing didn't beat the baseline on the pre-registered primary metric.** Standard augmentation
  improved macro F1, AUC and light-skin psoriasis recall, but missed the bar on balanced accuracy and needs
  {{aug_epochs}} epochs instead of {{base_epochs}}. Colour constancy and CLAHE made no clear difference.
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

- **Labels:** {{n_classes}} conditions with enough cases: {{class_list}}. Each case takes the dermatologists'
  top-weighted condition; cases whose top conditions tie across classes are dropped.
- **Splits at the case level** (a person's photos never appear in two splits; cases that share a photo are kept
  together): 80% for 5-fold cross-validation, 20% locked away as the test set, stratified by condition x skin
  tone. {{cv_cases}} CV cases ({{cv_images}} photos) and {{test_cases}} test cases ({{test_images}} photos).

{{class_counts_table}}

## Method

- **Model:** EfficientNet-B0 pretrained on ImageNet (torchvision), new 6-class head, class-weighted cross-entropy,
  AdamW, cosine schedule, 224x224 centre crops. No augmentation in the frozen method.
- **Every choice on CV only:** the epoch budget was picked on one CV fold; settings were compared with 5-fold CV
  and {{n_seeds}} seeds; the final setting was chosen with a rule written down before the results existed.
- **Test set scored once**, for the frozen method trained on all CV cases. A guard in `src/test_eval.py` refuses
  to score it again.
- **Uncertainty everywhere:** 95% bootstrap confidence intervals (resampling cases) for every metric, and a gap is
  called real only if its interval excludes zero. Metrics are case level: photo probabilities are averaged per case.

## Results

### Overall

{{overall_table}}

\* = outside the noise (95% CI excludes 0). Figures: [CV confusion matrix](docs/figures/baseline_confusion_matrix.png),
[test confusion matrix](docs/figures/test_confusion_matrix.png).

### Per condition (test set)

{{test_recall_table}}

Dermatitis, the most common class, is often confused with urticaria, insect bites and tinea.

### Per skin tone

Fitzpatrick (eFST) groups, the primary analysis:

{{fst_table}}

Monk (US) groups, a secondary check:

{{monk_table}}

Test results for small groups, or groups missing a condition, are marked as weak evidence. Figures:
[CV, 3 seeds](docs/figures/phase4_by_skin_tone_seeds.png), [test](docs/figures/test_by_skin_tone.png).

### Is the skin-tone gap real? (Phase 4)

In CV, FST III-IV beat FST I-II on balanced accuracy by {{p4_ba_gap}}. Balanced accuracy gives each condition
equal weight, so a few small conditions can drive it. Holding the condition fixed:

- Dermatitis only (enough cases in every group): {{p4_derm_gap}}
- Accuracy, crude: {{p4_crude_gap}}; with every group given the same condition mix: {{p4_std_gap}}
- Logistic regression `correct ~ tone + condition`: odds ratio {{p4_or}} for FST III-IV vs I-II; adding photo
  brightness, contrast, sharpness and resolution: {{p4_or_photo}}

The gap comes mainly from two small classes on light skin: psoriasis recall was {{p4_psoriasis_12}} cases in
FST I-II vs {{p4_psoriasis_34}} in III-IV, and tinea {{p4_tinea_12}} vs {{p4_tinea_34}}. On the test set the
FST III-IV minus I-II gap is {{test_fst_gap}}. Photo quality doesn't explain it. Figure:
[gap checks](docs/figures/phase4_gap_checks_efst.png).

### Image processing (Phase 5)

Four settings, {{n_seeds}} seeds x 5 folds each, the epoch budget re-picked per setting. Changes are paired
against the baseline on the same cases (percentage points, 95% CI; \* = outside the noise).

{{p5_table}}

The pre-registered rule (a setting must improve overall balanced accuracy beyond the noise without making any
FST group worse) keeps **{{p5_selected}}**. Augmentation is a documented secondary finding, not adopted:
better macro F1, AUC and light-skin psoriasis recall, but it missed the primary metric, needs {{aug_epochs}}
epochs instead of {{base_epochs}}, and tinea recall dropped (borderline). With {{p5_n_comparisons}} comparisons, an isolated
"outside the noise" result is weak evidence. Figure: [paired changes](docs/figures/phase5_paired_vs_baseline.png).

### Where the model looks (Grad-CAM, Phase 6)

Heatmaps from the last convolutional layer on all {{cam_n_images}} CV photos. On average only
{{cam_border_all}} of the heatmap falls in the outer 10% border, against {{cam_border_uniform}} if the model
looked everywhere equally: it mostly looks at the middle, where people put the lesion. Correctly predicted cases
put {{cam_border_correct}} of the heatmap on the border; wrongly predicted cases {{cam_border_incorrect}}, in every
FST group. In example grids, correct predictions sit on the plaque, bite or patch on light, medium and dark skin
alike, while wrong ones often land on clothing, background objects or blacked-out parts of the photo. No rulers
appeared in the cases inspected. The example grids show SCIN photos and are kept out of the repo; the
[mean heatmap per group](docs/figures/phase6_mean_gradcam.png) has no photos.

### How trustworthy are the probabilities?

On CV out-of-fold predictions, the expected calibration error (ECE) is {{cal_ece}}; its interval is lopsided
because resampling itself adds noise, which ECE counts as miscalibration. On average the model is {{cal_gap}}
overconfident, and in the middle of the range its probabilities match how often it is right.
**Very high probabilities are not reliable:** the {{cal_high_n}} CV cases with a top probability of 80% or more
averaged {{cal_high_conf}} confidence but were right only {{cal_high_acc}} of the time. Temperature scaling
(fitted per CV fold, T = {{cal_temps}}) didn't help (ECE change {{cal_temp_ece}}). Because of the class-weighted
loss, the probabilities behave as if all conditions were equally common: dermatitis gets {{cal_derm_pred}} of the
probability on average but is {{cal_derm_share}} of the cases. Figure: [reliability](docs/figures/calibration_cv.png).

## Limitations

- **Darker skin is barely represented** ({{fst56_cv_n}} CV and {{fst56_test_n}} test cases for FST V-VI), so
  performance there is unknown, not shown to be fine.
- **Only {{n_classes}} conditions.** For anything else, including healthy skin, the model still picks one of them.
- **Modest accuracy and small groups:** the test set has {{test_cases}} cases, so its intervals are wide.
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

Upload a photo to see the top guess, the probabilities for all {{n_classes}} conditions, a Grad-CAM overlay and a
reliability note that is the same for everyone (the app doesn't guess your skin tone). Below {{app_threshold}}
top probability it says "uncertain"; in CV that is {{app_share_uncertain}} of cases, and the top guess was right
{{app_acc_confident}} of the time above the threshold vs {{app_acc_uncertain}} below. Photos are processed in
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
