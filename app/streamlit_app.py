"""Streamlit demo: upload a skin photo, see the model's guess, its Grad-CAM and how reliable it is.

Educational demo, not a medical diagnosis. CPU inference with the frozen model
(models/final_baseline.pt). Every performance number comes from
reports/metrics.json. Uploaded photos are processed in memory and never saved.

Usage (from the repo root):
    streamlit run app/streamlit_app.py
    streamlit run app/streamlit_app.py -- --config configs/app.yaml
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config paths are relative to the repo root

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import UnidentifiedImageError  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from src.app_logic import (ci_text, confidence_state, format_probabilities, pct_text,  # noqa: E402
                           prepare_image, pretty, reliability_numbers)
from src.gradcam import GradCAM, overlay  # noqa: E402
from src.models import build_model  # noqa: E402
from src.transforms import eval_transform  # noqa: E402
from src.utils import load_config  # noqa: E402

BANNER = "Educational demo, not a medical diagnosis. Do not use to make health decisions."
SCIN_URL = "https://github.com/google-research-datasets/scin"
LICENSE_URL = "https://github.com/google-research-datasets/scin/blob/main/LICENSE"


def config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/app.yaml")
    return parser.parse_known_args(sys.argv[1:])[0].config


@st.cache_resource
def load_model(weights_path, n_classes, model_name, target_layer):
    """The model and its Grad-CAM, or None if the weights file is missing. CPU only."""
    path = Path(weights_path)
    if not path.exists():
        return None
    model = build_model(model_name, n_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, GradCAM(model, model.get_submodule(target_layer)), sha


def reliability_note(r, ci_level):
    """The same note for every upload: it does not depend on the photo."""
    st.subheader("How reliable is this model?")
    st.markdown(
        f"- **On {r['test']['n_cases']} held-out test cases** (scored once, at the end): balanced "
        f"accuracy {ci_text(r['test']['balanced_accuracy'], ci_level)}, AUC {ci_text(r['test']['macro_auc_ovr'], ci_level)}.\n"
        f"- **In cross-validation on {r['cv']['n_cases']} other cases:** balanced accuracy "
        f"{ci_text(r['cv']['balanced_accuracy'], ci_level)}, AUC {ci_text(r['cv']['macro_auc_ovr'], ci_level)}.\n"
        f"- Balanced accuracy is the average hit rate over the {len(r['classes'])} conditions; "
        f"guessing at random would score about {1 / len(r['classes']):.2f}.\n"
        f"- When the top guess had at least {r['confidence']['threshold']:.0%} probability, it was "
        f"right {pct_text(r['confidence']['accuracy_when_confident'], ci_level)} of the time in "
        f"cross-validation; below that, {pct_text(r['confidence']['accuracy_when_uncertain'], ci_level)}.")
    dark = r["fst_v_vi"]
    absent = f", and none with {', '.join(dark['classes_absent'])}" if dark["classes_absent"] else ""
    st.warning(
        f"**Darker skin is poorly tested.** The test set has only {dark['n_cases']} cases with "
        f"Fitzpatrick skin types V–VI{absent}. Balanced accuracy there was "
        f"{ci_text(dark['balanced_accuracy'], ci_level)}: so uncertain that the model's performance on darker "
        f"skin is essentially unknown. On the Monk skin-tone scale, the darkest group (Monk 7–10) has "
        f"{r['monk_7_10']['n_cases']} test cases, essentially none.\n\n"
        "This note is the same for every photo. The app does not try to judge your skin tone from "
        "the upload, because it can't do that reliably.")
    with st.expander("Test results per condition"):
        st.dataframe(pd.DataFrame([{"condition": row["class"], "test cases": row["cases"],
                                    "recognised": row["correct"],
                                    "share recognised": f"{row['correct'] / row['cases']:.0%}"}
                                   for row in r["test_recall_by_class"]]),
                     hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="Skin condition triage (demo)", page_icon="🩺")
    st.error(f"**{BANNER}**")
    st.title("Skin condition triage: educational demo")

    config = load_config(config_path())
    paths, app_cfg = config["paths"], config["app"]
    try:
        r = reliability_numbers(paths["metrics_json"])
    except (FileNotFoundError, KeyError) as err:
        st.error(f"The results file is incomplete, so the app can't show its reliability ({err}).")
        st.stop()
    classes = r["classes"]
    st.markdown(f"Upload a photo of a skin problem. The model picks one of **{len(classes)} "
                "conditions**: " + ", ".join(pretty(c) for c in classes) + ".")
    st.caption("This app processes your photo in memory and does not save it.")

    loaded = load_model(paths["final_model"], len(classes), config["model"]["name"],
                        config["gradcam"]["target_layer"])
    if loaded is None:
        st.warning(f"The model weights ({paths['final_model']}) are not on this machine, so the "
                   "app can't make predictions. They are kept out of git; see the README for how "
                   "to get them. The reliability note below still applies.")
    else:
        model, cam_fn, sha = loaded
        if sha != r["model_sha256"]:
            st.warning("These model weights are not the ones that were tested, so the numbers "
                       "below don't describe them.")
        upload = st.file_uploader("Photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
        if upload is not None:
            try:
                image = prepare_image(upload, config["images"]["cache_short_side"])
            except (UnidentifiedImageError, OSError):
                st.error("That file couldn't be read as an image. Please try a JPG or PNG photo.")
                st.stop()
            x = eval_transform(config["images"])(image).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].numpy()
            top = int(np.argmax(probs))
            cam, _ = cam_fn(x, top)
            size = config["images"]["input_size"]
            photo = np.asarray(T.Compose([T.Resize(size), T.CenterCrop(size)])(image))

            if confidence_state(probs, app_cfg["low_confidence_threshold"]) == "uncertain":
                threshold = app_cfg["low_confidence_threshold"]
                st.warning(f"**Uncertain.** The top guess, {pretty(classes[top])}, has only "
                           f"{probs[top]:.0%} probability, below the {threshold:.0%} the app "
                           "needs to call it a confident guess. Treat this as no answer.")
            else:
                st.info(f"**Top guess: {pretty(classes[top])}** ({probs[top]:.0%} probability).")
            st.caption(f"The model only knows these {len(classes)} conditions. For anything "
                       "else (or healthy skin) it will still pick one of them, and may be "
                       "confidently wrong.")

            st.subheader(f"Probabilities for all {len(classes)} conditions")
            rows = format_probabilities(probs, classes)
            st.dataframe(pd.DataFrame({"condition": [c for c, _, _ in rows],
                                       "probability": [100 * p for _, p, _ in rows]}),
                         column_config={"probability": st.column_config.ProgressColumn(
                             "probability", format="%.1f%%", min_value=0, max_value=100)},
                         hide_index=True, use_container_width=True)

            st.subheader("Where the model looked (Grad-CAM)")
            left, right = st.columns(2)
            left.image(photo, caption="What the model sees (centre crop)",
                       use_container_width=True)
            right.image(overlay(photo, cam, config["gradcam"]["overlay_alpha"]),
                        caption=f"Brighter = pushed the score for {pretty(classes[top])} up",
                        use_container_width=True)
            st.caption("A heatmap on the skin problem is a good sign; one on clothing, the "
                       "background or the photo's edges suggests the guess is unreliable.")

    reliability_note(r, config["evaluation"]["ci_level"])
    st.divider()
    st.caption(f"Trained on the [SCIN dataset]({SCIN_URL}) (Google Research and Stanford Medicine), "
               f"used under the [SCIN Data Use License]({LICENSE_URL}), provided as-is without "
               "warranty. Model: EfficientNet-B0 fine-tuned on SCIN photos.")


main()
