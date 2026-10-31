# Skin Condition Triage Across Skin Tones

## What this project is
A computer vision project that classifies common skin conditions from photos using a fine-tuned CNN, explains predictions with Grad-CAM, and reports how performance varies across skin tones. It ends in a Streamlit demo app.

This is a portfolio project for a Master's in AI application (NUS/NTU). The owner must be able to explain every part of it in an interview, so clarity beats cleverness.

## Hard rules
- Keep models simple: pretrained ResNet50 or EfficientNet-B0 via torchvision or timm. No transformers, GANs, or custom architectures.
- Use PyTorch, OpenCV, scikit-learn, pandas, matplotlib, Streamlit.
- Every script must run end to end from the command line with a config file (YAML), no hardcoded paths.
- Training must also run on a free GPU (Kaggle or Colab). Provide a notebook that clones the repo and calls the scripts.
- Set random seeds everywhere. Log every run's config and metrics to a `runs/` folder.
- Split data at the patient/contributor level, never at the image level, so the same person never appears in both train and test.
- Apply augmentation to the training split only.
- Ask the owner before any decision that changes scope, dataset, or labels.
- After finishing each phase, write a short plain-English explanation of what was built and why into `docs/learning-notes.md`.

## Evaluation rules (lessons from a previous project)
- Protect the test set. Choose every setting (augmentation, preprocessing, thresholds, model, epochs) using the validation set only. Score the test set once, at the end, for a frozen method.
- One source of truth for results: every reported number lives in `reports/metrics.json`. The README and the Streamlit app read from it. Never hand-copy or hardcode a metric.
- Never report a number that no code in the repo produces. If results conflict, investigate before updating anything.
- Report uncertainty: 95% bootstrap confidence intervals for test metrics, resampling patients (not images). Say plainly when a difference is within noise, including per-skin-tone differences with small groups.
- Report negative results honestly. If a technique doesn't help, document it rather than hiding it.
- Check labels and splits with tests: no patient in two splits, labels match the metadata, augmentation applied to training only.
- Standing permission: at the end of each phase, commit and push code, configs, tests and docs (including results files like `reports/metrics.json` and figures) without asking.
- Always ask the owner first before deleting files, committing data or images, or force-pushing.

## Data
- Primary dataset: SCIN (Google Research), https://github.com/google-research-datasets/scin. Read its README and data-use license before writing download code. Follow its official download instructions; do not guess file layouts.
- Use the dermatologist condition labels and the skin-tone labels it provides (estimated Fitzpatrick and Monk skin tone).
- Optional second test set: DermaCon-IN (Indian clinical images). Only add it if the owner confirms she has access.
- Do not commit any images to git. Add data folders to `.gitignore`.

## Phases
1. Setup and data exploration: repo structure, download script, EDA notebook (class counts, skin-tone distribution, image sizes, missing labels).
2. Label selection: pick 5-8 common conditions with enough images; build a clean metadata CSV with patient-level splits.
3. Preprocessing and baseline: image pipeline (resize, normalize), train baseline CNN, evaluate accuracy, balanced accuracy, macro F1, confusion matrix.
4. Skin-tone analysis: break down every metric by skin-tone group, with confidence intervals across 3+ seeds.
5. Image-processing experiments: compare no augmentation vs. standard augmentation vs. color constancy vs. CLAHE. Report results overall and per skin tone in one table.
6. Explainability: Grad-CAM heatmaps for correct and incorrect predictions across skin tones.
7. Streamlit app: upload image, show prediction with confidence, Grad-CAM overlay, and a reliability note for that skin-tone group. Show a clear banner: "Educational demo, not a medical diagnosis."
8. Write-up: README with problem, data, method, results tables, figures, limitations, and how to run.

## Repo layout
```
configs/        YAML configs per experiment
data/           raw/ and processed/ (gitignored)
notebooks/      EDA and Kaggle/Colab runner
src/            data.py, transforms.py, models.py, train.py, evaluate.py, gradcam.py
app/            streamlit_app.py
runs/           logs and metrics per run (gitignored except summaries)
docs/           learning-notes.md, figures
```
