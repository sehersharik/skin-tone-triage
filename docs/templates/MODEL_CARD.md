# Model card: skin-condition triage baseline (EfficientNet-B0 fine-tuned on SCIN)

<!-- Generated from docs/templates/MODEL_CARD.md by `python -m src.make_docs --config configs/app.yaml`.
     Every number comes from reports/metrics.json. Also used as the GitHub Release notes. -->

**Educational demo, not a medical device. Do not use it to make health decisions.**

## Model

- **File:** `final_baseline.pt` (PyTorch `state_dict`), SHA-256 `{{model_sha256}}`
- **Architecture:** EfficientNet-B0 (torchvision) with ImageNet weights, final layer replaced by a
  {{n_classes}}-class layer: {{class_list}}.
- **Training:** fine-tuned on all {{cv_cases}} cross-validation cases ({{cv_images}} photos) of the SCIN subset
  below, {{model_epochs}} epochs, seed {{model_seed}}, class-weighted cross-entropy, AdamW, cosine schedule,
  224x224 centre crops, no augmentation. Trained with PyTorch {{model_torch}}. Code and configs:
  <https://github.com/sehersharik/skin-tone-triage> (`configs/final.yaml`).
- **Input:** one RGB photo; resize the short side to 256 px, then 224 px, centre-crop 224x224, ImageNet
  normalisation. For a case with several photos, average the photos' probabilities.

## Intended use

- Teaching and portfolio demonstration of careful evaluation: per-skin-tone analysis with confidence intervals,
  a once-only test set, Grad-CAM.
- **Out of scope:** diagnosis, triage of real patients, any decision about health or treatment, use on
  conditions other than the {{n_classes}} above, and estimating anyone's skin tone.

## Performance

Test set: {{test_cases}} cases, scored once. 95% bootstrap CIs, resampling cases.

{{overall_table}}

Per Fitzpatrick (eFST) group:

{{fst_table}}

## Known limitations

- **Darker skin is essentially untested.** The test set has {{fst56_test_n}} FST V-VI cases (none with
  {{fst56_absent}}), where balanced accuracy was {{fst56_test_ba}}, and {{monk710_test_n}} Monk 7-10 cases.
- **Light-skin gap in cross-validation:** worse balanced accuracy on FST I-II than III-IV, mostly explained by
  the condition mix and by poor psoriasis and tinea recognition on light skin; not clearly seen on the test set.
- **Only {{n_classes}} conditions.** Any other condition, or healthy skin, is still assigned one of them.
- **Overconfident at the top:** in cross-validation, top probabilities of 80% or more were right
  {{cal_high_acc}} of the time. Probabilities behave as if all conditions were equally common (class-weighted
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
- **What was changed:** a subset of SCIN was used ({{n_classes}} conditions; cases with a clear top label),
  photos were resized, and the model above was trained on them. No SCIN images are distributed with this model.
- **No re-identification:** you may not attempt to re-identify or re-link any person whose data was used to
  create SCIN. This is a condition of the SCIN license.
- **No warranty:** the SCIN data, and this model, are provided as-is, without warranties of any kind, including
  accuracy and fitness for a particular purpose.
- **No endorsement:** this model is not connected with, sponsored or endorsed by Google, Stanford Medicine or the
  SCIN creators.
