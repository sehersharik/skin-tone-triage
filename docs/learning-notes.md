# Learning notes

Plain-English notes on what was built in each phase and why.

## Phase 1: Setup and data exploration

**What was built**

- A repo layout that separates configs, source code, notebooks, data and run logs.
- `src/download_scin.py`: downloads SCIN from Google's public cloud bucket. It follows the official demo notebook
  (same bucket, same file paths) instead of guessing file names. The list of images to fetch comes from
  `scin_cases.csv`, and every run saves its config and a summary to `runs/`. It can fetch all images, a seeded
  random sample, or only the CSVs, because the full set is 12.6 GB.
- `src/data.py`: small functions that turn the raw CSV columns into things we can count, e.g. the top condition
  of a case and a single estimated Fitzpatrick value per case. They are unit-tested in `tests/test_data.py`.
- `notebooks/01_eda.ipynb`: counts of cases, images, labels and skin tones, plus image sizes.

**Why it was built this way**

- *Cases, not images, are the unit.* One person sends 1-3 photos of the same problem. If photos of the same case
  ended up in both training and test, the model could memorise the person's skin instead of learning the
  condition, and the test score would be too optimistic. So every count is reported per case as well as per image.
- *Labels are probabilities, not single answers.* Up to 3 dermatologists each gave up to 3 guesses. SCIN merges
  these into weights such as `{'Eczema': 0.69, 'Psoriasis': 0.31}`. To train a single-label classifier we take the
  top-weighted condition, and we flag cases where two conditions tie because there is no clear answer for them.
- *Skin tone has three sources.* Self-reported Fitzpatrick (often missing), dermatologist-estimated Fitzpatrick,
  and Monk skin tone estimated by lay graders in the US and in India. They do not always agree, which is itself
  worth reporting: if the group label is noisy, per-group results are noisier still.

**What we learned**

- Only about 40% of cases have a usable single condition label; many photos were judged not gradable, and
  nearly a third of the labelled cases have a tie at the top.
- The data is dominated by lighter skin tones. Darker groups (FST V-VI, Monk 7-10) are small for every condition,
  so per-skin-tone results for those groups will have wide confidence intervals.
- 15 image files are shared between 29 cases. Those cases must stay together in one split.
- Images are PNG, mostly 810x1080 or 1080x810 (portrait or landscape phone photos).

## Phase 2: Label selection and splits

**What was built**

- `configs/labels.yaml`: the six classes and which SCIN condition names map to each. Eczema and allergic contact
  dermatitis are merged into one *dermatitis* class, because dermatologists often list both for the same photo.
- `src/build_metadata.py`: turns the raw SCIN CSVs into `data/processed/cases.csv`, one row per case with its
  class, skin-tone groups, label weights, image paths, split (`cv` or `test`) and CV fold.
- `tests/test_splits.py`: checks the rules below on the real output.

**Why it was built this way**

- *One class per case.* We take the condition(s) with the top weight. A tie is fine if both tied conditions land in
  the same class (e.g. eczema vs. allergic contact dermatitis); otherwise the label is ambiguous and the case is dropped.
  We don't drop low-confidence labels, but we store the weight so we can test later whether they hurt.
- *Groups, not just cases.* A few photos appear in more than one case. Cases that share any photo are joined into
  a group (using union-find: keep merging until no shared photo links two groups). Splits are made per group,
  so the same photo can never be in training and test.
- *Stratified by class x skin tone.* The darker-skin groups are small. Stratifying makes each split get its fair
  share of them rather than leaving it to chance.
- *Test set used once.* 20% of cases are locked away. All tuning uses 5-fold cross-validation on the other 80%:
  train on 4 folds, check on the 5th, rotate. Only the final, frozen method is scored on the test set.
- *Case-level evaluation.* A case has 1-3 photos of the same problem. The model predicts each photo, the
  probabilities are averaged per case, and metrics are computed per case, so a case with 3 photos doesn't count 3 times.

**What we learned**

- 1,161 cases (2,538 images) remain. Dermatitis alone is over half of them, so balanced accuracy and macro F1
  matter more than plain accuracy.
- The FST V-VI test group is very small (single digits for most classes). Per-tone results for that group will
  be reported with wide confidence intervals and should be read as indicative only.

## Phase 3: Preprocessing and baseline

**What was built**

- `src/prepare_images.py`: resizes every image once (short side 256 px) and caches it, because decoding full-size
  phone photos every epoch is slow on Kaggle's few CPU cores.
- `src/transforms.py`: resize the short side to 224, centre-crop to 224x224, normalise with ImageNet mean/std.
  No augmentation yet; Phase 5 compares augmentation options. Validation always uses the plain transform.
- `src/models.py`: a pretrained EfficientNet-B0 (ImageNet weights) with its last layer replaced by a 6-class layer.
- `src/train.py`: 5-fold cross-validation on the CV pool. For each fold, train on the other 4 folds, predict the
  held-out fold, average the photo probabilities within each case. Every case gets exactly one *out-of-fold*
  prediction, from a model that never saw it. The test set is dropped before training starts.
- `src/evaluate.py`: case-level balanced accuracy, macro F1 and macro one-vs-rest AUC, overall and per skin-tone
  group, with 95% bootstrap confidence intervals. Everything is written to `reports/metrics.json` (`baseline_cv`)
  and the figures are drawn from that file.
- `notebooks/kaggle_colab_runner.ipynb`: clones the repo on Kaggle and runs the scripts on a free T4 GPU. Each
  finished fold is saved straight away, so a timeout doesn't lose work.

**Why it was built this way**

- *Class-weighted loss.* Dermatitis is over half the data. Without weights the model can score well by mostly
  saying "dermatitis". Each class's loss is scaled by the inverse of its frequency so rare classes count as much.
- *Balanced accuracy, not accuracy.* Balanced accuracy is the average recall over the classes, so every class
  counts equally. Plain accuracy is reported only for context; with weighted training it can be lower than
  always guessing the majority class, and that is expected.
- *A fixed epoch budget instead of early stopping.* Early stopping on the fold being scored would quietly tune on
  it. Instead, 5, 10, 15 and 20 epochs were each tried once, scored on fold 1 (`cv_fold 0`), and the best (5) was
  then used unchanged for all 5 folds.
- *Bootstrap confidence intervals.* The cases are resampled with replacement 2,000 times and the metric recomputed
  each time; the middle 95% of those values is the interval. A gap between two skin-tone groups counts as real only
  if the interval of the *difference* excludes zero.

**Caveats**

- Choosing the epoch budget on fold 1 makes fold 1's out-of-fold score slightly optimistic, because that fold
  helped pick the setting. The effect is small (one choice out of four candidates), and folds 2-5 played no part
  in the choice. The choice itself rests on one fold, so "5 epochs" is a reasonable setting, not a proven optimum.
- One seed only. Phase 4 repeats the run with 3 seeds to see how much the numbers move.

**What we learned** (numbers in `reports/metrics.json`)

- The baseline is clearly better than chance but weak. It does best on dermatitis, urticaria and folliculitis and
  worst on psoriasis and tinea, the two smallest classes.
- Longer training scored worse on the budget fold: without augmentation the model starts memorising its training
  photos within a few epochs. This is a reason to test augmentation in Phase 5.
- Surprisingly, the model did worse on the lightest skin group (FST I-II) than on FST III-IV, and that gap is
  outside the bootstrap noise. The darkest group (FST V-VI) is too small to say anything firm. The groups also have
  different mixes of conditions, so Phase 4 checks whether the gap is about skin tone or about condition mix.
