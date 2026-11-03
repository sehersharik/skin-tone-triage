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

## Phase 4: Is the skin-tone gap real?

**What was built**

- `configs/seeds.yaml` + `src/train.py`: the Phase 3 baseline rerun with 3 seeds (42, 43, 44) on the same folds,
  with the 5-epoch budget fixed. The seed changes the random parts (the new last layer's starting weights and the
  order of training batches), so 3 seeds show how much the results move by chance alone.
- `src/image_factors.py`: brightness, contrast, sharpness and resolution of every CV photo.
- `src/tone_analysis.py`: four ways of asking whether skin tone matters once the condition is taken into account.
  Results are in `reports/metrics.json` (`tone_analysis_cv`) and the `phase4_*` figures.

**Why it was built this way**

- *Seeds.* One training run can be lucky or unlucky. Each group's metric is reported as the mean over seeds, the
  spread between seeds, and a bootstrap confidence interval.
- *Condition mix.* The skin-tone groups have different mixes of conditions, and some conditions are harder than
  others. A group could look worse just because it has more of the hard conditions. Four checks separate the two:
  1. *One condition only* (dermatitis, the only class with enough cases in every group): the mix can't differ.
  2. *Same-mix accuracy*: each group's per-condition recalls are reweighted to the same condition mix (the pooled
     one), like age-standardised death rates in epidemiology.
  3. *Logistic regression*: `correct ~ tone group + condition`. The tone odds ratio says how much more likely a
     correct prediction is in a group compared with FST I-II, *for the same condition*. 1 means no difference.
  4. *Photo factors*: added to the regression to see whether photo quality explains the gap.
- *Balanced accuracy is already mix-free in one sense*: it is the plain average of the per-condition recalls, so
  the mix of cases doesn't change it. What it does do is give each condition equal weight, so a small condition
  (18 psoriasis cases) moves it as much as dermatitis (230 cases). Same-mix accuracy weights conditions by how
  common they are, so comparing the two shows whether a gap comes from common or rare conditions.

**What we learned**

- Seed 42 reproduced Phase 3 exactly, so the pipeline is deterministic. Between seeds, the scores move by a few
  points, and more for the smaller groups: a single seed can mislead.
- The FST I-II vs III-IV balanced-accuracy gap holds across 3 seeds. But once the condition is held fixed, most
  of it goes away: dermatitis recall is about the same in both groups, and the same-mix accuracy gap and the
  regression's tone effect are about half the size, with confidence intervals that touch or cross "no difference".
- The balanced-accuracy gap comes mostly from two small classes in FST I-II: psoriasis (almost never recognised)
  and tinea. These are only about 20 cases each. That is a condition-specific weakness worth reporting, not
  evidence that the model generally works worse on light skin.
- The direction is the same in every check (FST I-II never comes out ahead), so a small real disadvantage can't be
  ruled out. The Monk grouping tells the same story, with the condition-adjusted gap just outside the noise.
- Photo quality doesn't explain it. FST I-II and III-IV photos have the same brightness, sharpness and resolution,
  and only slightly different contrast. Adding these factors to the regression doesn't shrink the tone effect.
  Photos of darker skin are darker overall (expected, since the skin is in the photo) and, unexpectedly, sharper.
- The darkest groups (FST V-VI, Monk 7-10) show no sign of doing worse, but with 61 and 32 cases the confidence
  intervals are too wide to rule out a large gap in either direction. That is a limit of the data, not a finding.

## Phase 5: Image-processing experiments

**What was built**

- `src/transforms.py`: two kinds of image processing, kept separate.
  - *Preprocessing* is the same every time and is applied to every image, in training and evaluation, so the
    model always sees images prepared the same way. **Shades of Gray** estimates the colour of the light (a
    "power mean" of each colour channel) and rescales the channels so that light becomes neutral grey, which
    removes colour casts from different phones and lamps. **CLAHE** boosts local contrast tile by tile, on the
    lightness channel only so colours don't shift, with a clip limit so noise isn't amplified.
  - *Augmentation* is random and used on training images only: small rotations, random crops, flips and mild
    colour jitter. Each epoch the model sees slightly different versions of each photo, which makes memorising
    them harder.
- `configs/phase5/`: one shared base config and one small file per setting, so it is obvious that the four
  settings differ in exactly one thing. Each setting changes one thing against the baseline (no combinations).
- `src/compare_settings.py`: scores every setting on the same cases and compares it with the baseline using a
  *paired* bootstrap: each resample of cases is scored under every setting, so differences between cases cancel
  out and only the effect of the setting is left. That makes small real differences easier to see.

**Why it was built this way**

- *The epoch budget was re-picked for every setting* (on fold 1, seed 42, as in Phase 3). Augmentation changes
  how fast a model overfits, so judging every setting at 5 epochs would have been unfair to augmentation.
- *The rule for choosing the final setting was written into the config before any result existed*: a setting
  wins only if it improves overall balanced accuracy beyond the noise and makes no skin-tone group worse beyond
  the noise; otherwise the simplest setting (the baseline) stays. Fixing the rule first stops us from picking
  whichever number happens to look best afterwards.
- *Many comparisons* (60) are made, so about 3 can look "outside the noise" by chance alone. One isolated
  result, especially in a small group, is weak evidence; a pattern across measures is stronger.

**What we learned** (numbers in `reports/metrics.json`, `phase5_cv`)

- **The rule keeps the baseline.** No setting improved overall balanced accuracy beyond the noise. This is a
  negative result and is reported as such.
- **Augmentation came closest and helped elsewhere.** Its overall balanced-accuracy gain just misses the bar, but
  macro F1 and macro AUC improve beyond the noise, the three seeds agree with each other much more closely, and
  the budget search shows it no longer overfits after 5 epochs (it chose 15). Most strikingly, it fixes part of
  the Phase 4 weakness: psoriasis on FST I-II goes from almost never recognised to recognised about a fifth of the
  time, which narrows the light-skin gap. The price: tinea recall tends to drop (borderline), and training takes
  about three times longer.
- **Shades of Gray and CLAHE made no clear difference.** CLAHE scored worse on FST V-VI beyond the noise, but
  that is one result out of 60, on 61 cases, so it is weak evidence rather than a finding.
- Kaggle updated its PyTorch build between Phase 4 and Phase 5 (CUDA 12.8 to 13.0). The baseline rerun therefore
  differs from Phase 4 in the 7th decimal place of some probabilities: every predicted class and every metric is
  the same. Bit-for-bit reproducibility needs the same software as well as the same seed.

**Decision**

- The baseline is the frozen method, exactly as the pre-registered rule says. Augmentation is reported as a
  secondary finding (better macro F1, AUC and light-skin psoriasis recall; missed the primary metric; 3x the
  training time; tinea recall dropped), not adopted. Switching to it after seeing the results would be the kind
  of choice the rule exists to prevent.
- Workflow lesson: Kaggle's "Save to GitHub" sync committed the executed notebook to `main` and caused a push
  conflict. Results now come back only as the downloaded zip.

## Phase 6: Explainability (Grad-CAM)

**What was built**

- `src/train_final.py` + `configs/final.yaml`: the frozen method (the baseline, 5 epochs, seed 42) trained once on
  all 928 CV cases. The weights are in `models/`, kept out of git.
- `src/gradcam.py`: Grad-CAM on the last convolutional layer. For a chosen class it asks which parts of the
  7x7 feature map raised that class's score, and upsamples the answer to the photo. Bright = pushed the score up.
- Grids (in `reports/gradcam/`, not in git because they show patient photos): random cases per skin-tone group,
  correct and incorrect, one per condition and two each for psoriasis and tinea. Wrong predictions show two
  heatmaps: toward the wrong answer and toward the true condition.
- A photo-free check committed to the repo: the average heatmap per FST group (`docs/figures/phase6_mean_gradcam.png`)
  and the *border share*, the share of each heatmap that falls in the outer 10% of the image
  (`phase6_gradcam` in `reports/metrics.json`). If the model looked everywhere equally, that share would be about
  35%.

**Why it was built this way**

- *Correct vs incorrect comes from out-of-fold predictions*, made by models that never saw the case. The heatmaps
  come from the final model, which *did* train on these cases, so they show where the frozen model looks, not
  how well it generalises. The test set is still untouched.
- *Random picks, not the clearest examples*, so the grids aren't flattering by design.
- *A number, not just pictures.* Looking at 44 heatmaps is easy to over-read. The border share is measured on all
  2,054 CV images, so it can confirm or contradict what the grids seem to show.

**What the model seems to focus on**

When the cross-validation models got a case right, the final model's heatmap usually sits on the affected skin:
the psoriasis plaque, the bite mark, the edge of a tinea patch, the cluster of folliculitis spots. For widespread
conditions like dermatitis and urticaria it spreads along the limb. This looks much the same on light, medium and
dark skin, and the average heatmap is a central blob in all three FST groups. When the prediction was wrong, the
heatmap toward the wrong answer often sits away from the lesion. In the grids it lands on **clothing** (a pink
strap, a grey T-shirt held up by the person, a black top), on **background objects** (blue toys, floor tiles, a
bright door frame, a blanket) and on **image edges and black blank areas** where part of the photo is blacked
out. The measurements agree: incorrectly predicted cases have clearly more heatmap mass in the border than correct
ones, in every FST group. Overall the border share is well below what even attention would give, so the model
mostly looks at the middle of the photo, where people usually put the lesion. **No rulers appeared in the cases
shown** (these are consumer phone photos), but 44 cases can't rule them out. Three cautions: a 7x7 map is coarse (each cell covers about 32
pixels), so "on the lesion" means "in that region"; a central focus partly reflects how people frame photos;
and the border share can't detect clothing or background in the middle of the image. The fair summary is that
the model mostly uses the skin, and when it is wrong, background and clothing are a likely part of why.

## Phase 7: The one-time test-set evaluation

**What was built**

- `src/test_eval.py` + `configs/test.yaml`: scores the frozen method (`models/final_baseline.pt`, the baseline
  trained on all 928 CV cases) on the 233 test cases, case level, with the same metrics as the CV evaluation,
  overall and per FST and Monk group, 95% bootstrap CIs (cases resampled), and a test-vs-CV comparison. Results
  are in `reports/metrics.json` (`test_final`) plus two figures.
- A guard: the script refuses to run once a test result exists, unless it is given `--overwrite-test-result`,
  so the test set can't be reused by accident.
- Tests: the guard; scoring twice on synthetic data gives identical numbers; and re-scoring the saved real
  test predictions reproduces `metrics.json` exactly.

**Why it was built this way**

- *The test set is scored once, for a method frozen in advance.* Every choice (model, image processing, epochs)
  was made on CV results. Looking at test results and then changing something would turn the test set into a
  second validation set, and its numbers would be optimistic again.
- *The code was committed before the scoring*, and the result records that commit, the model file's SHA-256
  fingerprint and the software versions. Anyone can check which code and which model produced the numbers.
- *Run on a laptop CPU.* 233 cases take about a minute, and CPU arithmetic gives exactly the same numbers on a
  rerun.

**What we learned** (numbers in `reports/metrics.json`, `test_final`)

- **The test numbers agree with cross-validation.** Test balanced accuracy and AUC are a little higher than the
  seed-42 CV values, but every test-minus-CV difference is within the noise. A small rise is expected: the final
  model trained on all CV cases, a quarter more data than each CV model. The test set has only 233 cases, so
  its confidence intervals are wide.
- The per-condition pattern matches CV: urticaria and insect bites are recognised best, psoriasis and tinea worst,
  and dermatitis is often confused with urticaria, insect bites and tinea.
- **The FST I-II vs III-IV gap seen in CV does not show up clearly on the test set**: the difference is within
  the noise, which fits the Phase 4 conclusion that it was mostly condition-specific and borderline.
- **FST V-VI scored much lower on the test set, but this is weak evidence.** It has only 15 test cases and no
  folliculitis, and its confidence interval runs from very poor to above average. It can't be dismissed either:
  it is one more reason the app's reliability note for darker skin must say plainly that the model is untested
  there.
- Monk 7-10 has only 3 test cases, all dermatitis: no conclusion is possible, and AUC can't even be computed.

## Phase 8: Streamlit demo app

**What was built**

- `app/streamlit_app.py`: upload a photo; see the top guess, the probabilities for all 6 conditions, a Grad-CAM
  overlay and a reliability note. The page logic that can be tested lives in `src/app_logic.py`.
- `src/app_confidence.py`: measures the app's "uncertain" rule on the CV out-of-fold predictions.

**Why it was built this way**

- *Every number comes from `reports/metrics.json`.* A test fails if a decimal, a percentage or a "6 conditions"
  is typed into the page code, so the app can never drift away from the results.
- *The same reliability note for everyone.* The app does not estimate skin tone from the photo: that would be
  unreliable, and a wrong "your skin type is..." would be worse than none. Instead it states plainly that darker
  skin is poorly tested.
- *An "uncertain" state.* Below 50% top probability the app calls the result uncertain. 50% is a round number,
  not tuned; how it behaves was measured on CV cases (never the test set): more than half of the cases fall below
  it, and the model is right noticeably more often above it than below. A model that only knows 6 conditions will
  still pick one for anything else, so the page says that too.
- *The upload is prepared exactly like the training images*: fix the phone's rotation, resize the short side to
  256 px, centre-crop 224x224. A demo that preprocesses differently from training quietly gets worse.
- *Privacy:* photos are processed in memory and never written to disk; Streamlit's usage statistics are off.

**What we learned**

- Running the real app found a bug the tests had missed: once Grad-CAM was attached, a plain prediction without
  gradients crashed. Phase 6 always ran with gradients, so its results were not affected. A test now covers it.
- A photo the model was trained on gets a near-100% answer. That shows the page works, not that the model is
  accurate; only the test-set numbers say that.

## Phase 9: Calibration check, deployment prep and the write-up

**What was built**

- `src/calibration.py`: checks whether the model's probabilities can be taken at face value, on CV predictions
  only. A *reliability table* groups cases by the model's top probability and compares it with how often the model
  was actually right; *ECE* averages that mismatch; the *Brier score* measures probability accuracy overall.
  *Temperature scaling* (dividing the model's log-probabilities by a number T, which softens or sharpens them) was
  fitted per CV fold so no case was scaled by a T fitted on itself.
- Deployment prep, inactive until the owner publishes the weights: `app/requirements.txt` (CPU-only PyTorch),
  a download that only keeps the file whose SHA-256 matches the tested weights, `docs/DEPLOY.md`, and the model
  card `docs/MODEL_CARD.md`.
- `src/make_docs.py`: the README and model card are generated from templates in `docs/templates/`, with every
  number filled in from `reports/metrics.json`. A test fails if they drift apart.

**What we learned**

- Overall the probabilities are close to honest (slightly overconfident, below the bar set in advance for "clearly
  overconfident"), but very high probabilities are not: cases above 80% were right only about half the time.
  Temperature scaling didn't help, because one global T can't fix just the top end.
- The class-weighted loss makes the probabilities behave as if all conditions were equally common, so the
  model gives dermatitis much less probability than its real share. That is a side effect of choosing balanced
  accuracy as the target, and worth saying out loud.

## Interview preparation: the 5 questions you're most likely to get

Quote numbers from the README tables (they come from `reports/metrics.json`), always with their interval.

**1. "Your accuracy is low. Is the model any good?"**
Start with what the number means: 6 classes, so guessing at random gives about 0.17 balanced accuracy; the model
is clearly above that on the test set, and the test result agrees with cross-validation. Then say why it isn't
higher: consumer phone photos (not clinic images), labels that are dermatologists' best guesses from photos alone,
heavy class imbalance, and a deliberately simple model. Finish with the point of the project: "I optimised for
an honest, reproducible number with error bars, not for a high one. A modest result you can trust is more useful
than an impressive one you can't."

**2. "How do you know you didn't overfit to the test set?"**
Walk through the safeguards in order: splits by case (with cases sharing a photo kept together) so no person is in
two splits; a fingerprint check that the splits never changed; every choice (epochs, image processing, final
setting) made with cross-validation only; a selection rule written down before Phase 5 results existed; the
evaluation code committed before scoring; and a guard that refuses to score the test set twice. The evidence that
it worked: test and CV agree within the noise.

**3. "Does it work worse on darker skin?"**
Be precise, because this is the heart of the project. "In cross-validation it actually looked worse on *lighter*
skin. When I held the condition fixed (dermatitis only, standardising the condition mix, and a regression with
condition as a covariate) most of that gap disappeared; what was left came mainly from psoriasis and tinea on light
skin, and it didn't clearly reappear on the test set. For the darkest skin (FST V-VI) there are too few cases to
say anything: the test interval runs from very poor to above average. So the honest answer is 'unknown', and the app
says so to every user rather than guessing their skin tone." Then say what you'd do: external data with more
darker skin, such as DermaCon-IN.

**4. "Augmentation improved F1 and AUC. Why didn't you use it?"**
"Because I'd decided the rule before seeing the results: improve overall balanced accuracy beyond the noise
without hurting any skin-tone group. Augmentation missed that bar. With about 60 comparisons, some will look
significant by chance, and switching to whichever metric happens to look good is exactly how results become
over-optimistic. I report it as a promising secondary finding, and the right next step is a pre-registered
confirmation run, not a quiet switch."

**5. "What does Grad-CAM show, and can you trust it?"**
Explain it in one breath: "It takes the gradient of the predicted class's score with respect to the last
convolutional feature maps, weights each map by its average gradient, and shows where those maps were active."
Then the finding: when the model is right, it mostly looks at the lesion, on all skin tones; when it is wrong, it
often looks at clothing, backgrounds or blacked-out areas, and a measured border share confirms that wrong cases
look at the edges more. Then the limits, which interviewers want to hear: the map is coarse (7x7), it shows where
the model looked, not why it was right, and the heatmaps came from the final model, which had seen those photos.
