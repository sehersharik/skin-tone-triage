# Deploying the demo app (Streamlit Community Cloud)

Nothing here has been published. The app's weights download is switched off
(`weights_url: null` in `configs/app.yaml`) until the owner creates the Release.

## What is already in place

- `app/requirements.txt`: lean, CPU-only dependencies (PyTorch from the CPU-only index, so no GPU
  libraries are installed). Community Cloud uses the requirements file next to the entry point.
- `.streamlit/config.toml`: 10 MB upload limit, usage statistics off.
- `src/app_logic.ensure_weights`: if `models/final_baseline.pt` is missing and `weights_url` is set, the
  app downloads it once and keeps it only if its SHA-256 equals `test_final.model_sha256` in
  `reports/metrics.json`, i.e. exactly the weights that were scored on the test set.
- `docs/MODEL_CARD.md`: the Release notes / model card (attribution, license, limitations).

## Steps (owner only)

1. **Publish the weights as a Release asset** on this repo, with the model card as the notes:

   ```bash
   gh release create model-v1 models/final_baseline.pt \
     --title "Frozen baseline model (EfficientNet-B0 fine-tuned on SCIN)" \
     --notes-file docs/MODEL_CARD.md
   ```

   A Release on a public repo is public: anyone can download the file.

2. **Point the app at it** in `configs/app.yaml`, then commit and push:

   ```yaml
   weights_url: https://github.com/sehersharik/skin-tone-triage/releases/download/model-v1/final_baseline.pt
   ```

3. **Create the app** at <https://share.streamlit.io> → *Create app* → *Deploy a public app from GitHub*:
   - Repository `sehersharik/skin-tone-triage`, branch `main`
   - Main file path `app/streamlit_app.py`
   - *Advanced settings* → Python **3.12** (the version the app was tested with). No secrets are needed.

4. **Check the live app**: banner at the top; upload a photo; the reliability note matches
   `reports/metrics.json`. If the weights fail the SHA-256 check, the app says so and does not predict.

## Privacy note for the hosted version

Uploaded photos are sent to Streamlit's servers, where this app processes them in memory and does not
save them. The app says "This app processes your photo in memory and does not save it", which is
accurate for the app itself; it can't make promises about the hosting platform.
