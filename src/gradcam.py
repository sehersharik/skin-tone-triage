"""Phase 6: Grad-CAM heatmaps from the final baseline model.

Grad-CAM (Selvaraju et al., 2017) asks: which parts of the last convolutional
feature map pushed the score of one class up? It takes the gradient of that
class score with respect to each of the 1280 feature maps (7x7), averages each
gradient map into one weight per feature map, adds the feature maps up with
those weights, keeps the positive part, and upsamples the 7x7 result to the
224x224 image. Bright = this region raised the class score.

Which cases count as "correct" or "incorrect" comes from the seed-42
OUT-OF-FOLD predictions: each case was predicted by a CV model that never saw
it. The heatmaps themselves come from the final model, which was trained on
all CV cases, including these. They illustrate where the frozen model looks;
they say nothing about how well it generalises. The test set is never loaded.

Outputs:
- grids per FST group x correct/incorrect -> `paths.gradcam_dir` (gitignored:
  they show SCIN photos)
- the mean heatmap per FST group (no photos) -> docs/figures/phase6_mean_gradcam.png
- how much heatmap mass falls on the image border, per group ->
  reports/metrics.json (`phase6_gradcam`)

Usage:
    python -m src.gradcam --config configs/final.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from src.data import case_images
from src.evaluate import EFST_ORDER, ci
from src.models import build_model
from src.prepare_images import cache_path
from src.train import get_device, load_cv_pool
from src.transforms import eval_transform
from src.utils import load_config, set_seed

CMAP = "inferno"   # sequential, perceptually uniform: dark = no influence, bright = most


class GradCAM:
    """Grad-CAM for one layer of a model. Call with a (1, 3, H, W) input and a class index."""

    def __init__(self, model, layer):
        self.model = model.eval()
        layer.register_forward_hook(self._keep_activations)

    def _keep_activations(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:   # not under torch.no_grad(), e.g. a plain prediction
            output.register_hook(self._keep_gradients)

    def _keep_gradients(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx):
        """Returns (heatmap in [0, 1] at the input's size, softmax probabilities)."""
        with torch.enable_grad():
            self.model.zero_grad()
            logits = self.model(x)
            logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)        # one weight per feature map
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.detach().cpu().numpy(), torch.softmax(logits, dim=1)[0].detach().cpu().numpy()


def border_share(cam, fraction):
    """Share of the heatmap's total mass in the outer `fraction` of the image on each side."""
    h, w = cam.shape
    dh, dw = round(h * fraction), round(w * fraction)
    inner = cam[dh:h - dh, dw:w - dw].sum()
    total = cam.sum()
    return float((total - inner) / total) if total > 0 else np.nan


def peak_in_border(cam, fraction):
    h, w = cam.shape
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    dh, dw = round(h * fraction), round(w * fraction)
    return bool(y < dh or y >= h - dh or x < dw or x >= w - dw)


def uniform_border_share(size, fraction):
    return border_share(np.ones((size, size)), fraction)


def select_cases(oof, classes, gcfg, rng):
    """Per FST group x correct/incorrect x class: a seeded random pick of cases (not the most
    confident ones, to avoid showing only flattering examples). Returns (picks, missing)."""
    picks, missing = [], []
    for group in EFST_ORDER:
        for correct in (True, False):
            for cls in classes:
                n = gcfg["cases_per_focus_class"] if cls in gcfg["focus_classes"] \
                    else gcfg["cases_per_class"]
                pool = oof[(oof.efst_group == group) & (oof.label == cls) & (oof.correct == correct)]
                if pool.empty:
                    missing.append({"efst_group": group, "correct": correct, "label": cls})
                    continue
                chosen = pool.iloc[np.sort(rng.choice(len(pool), min(n, len(pool)), replace=False))]
                picks.append(chosen)
    return pd.concat(picks, ignore_index=True), missing


def overlay(photo, cam, alpha):
    heat = plt.get_cmap(CMAP)(cam)[..., :3]
    return np.clip((1 - alpha) * photo / 255 + alpha * heat, 0, 1)


def slug(group):
    return group.replace(" ", "_").replace("-", "_")


def plot_grid(rows, missing, title, alpha, path):
    """One row per case: photo | heatmap toward the CV-predicted class | (if wrong) toward the true class."""
    incorrect = not rows.empty and not rows.correct.iloc[0]
    n_cols = 3 if incorrect else 2
    n_rows = max(len(rows) + (1 if missing else 0), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols + 0.6, 2.45 * n_rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, r in enumerate(rows.itertuples()):
        axes[i, 0].imshow(r.photo)
        axes[i, 0].set_title(f"true: {r.label}\nCV pred: {r.pred} ({r.pred_prob:.0%}); "
                             f"final: {r.final_pred}", fontsize=7.5)
        axes[i, 1].imshow(overlay(r.photo, r.cam_pred, alpha))
        border = "no positive evidence" if np.isnan(r.border_pred) else \
            f"border share {r.border_pred:.0%}"
        axes[i, 1].set_title(f"toward {r.pred}\n{border}", fontsize=7.5)
        if incorrect:
            axes[i, 2].imshow(overlay(r.photo, r.cam_true, alpha))
            axes[i, 2].set_title(f"toward {r.label} (true)", fontsize=7.5)
    if missing:
        axes[-1, 0].text(0, 0.5, "No such case for: " + ", ".join(m["label"] for m in missing),
                         fontsize=8, transform=axes[-1, 0].transAxes, va="center")
    fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), ax=axes, shrink=0.25, pad=0.02,
                 label="Grad-CAM (0 = no influence, 1 = most)")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def plot_mean_cams(mean_cams, counts, fraction, path):
    fig, axes = plt.subplots(1, len(mean_cams), figsize=(3.3 * len(mean_cams), 3.6))
    size = next(iter(mean_cams.values())).shape[0]
    d = round(size * fraction)
    for ax, (group, cam) in zip(np.atleast_1d(axes), mean_cams.items()):
        im = ax.imshow(cam, cmap=CMAP, vmin=0, vmax=1)
        ax.add_patch(plt.Rectangle((d, d), size - 2 * d, size - 2 * d, fill=False, ec="white",
                                   lw=1, ls="--"))
        ax.set_title(f"{group} (n={counts[group]} images)", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean Grad-CAM")
    fig.suptitle("Mean Grad-CAM toward the CV-predicted class, final model "
                 f"(dashed box: inside = not the outer {fraction:.0%} border)", fontsize=9)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def mean_ci(values, n_bootstrap, ci_level, rng):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    low, high = ci(boot, ci_level)
    return {"n_cases": int(len(values)), "mean": round(float(values.mean()), 4),
            "ci_low": low, "ci_high": high}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths, gcfg, ev = config["paths"], config["gradcam"], config["evaluation"]
    set_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])

    info = json.loads(Path(paths["final_model_info"]).read_text())
    classes = info["classes"]
    device = get_device(config["train"]["device"])
    model = build_model(config["model"]["name"], len(classes), pretrained=False)
    model.load_state_dict(torch.load(paths["final_model"], map_location="cpu"))
    model.to(device)
    cam_fn = GradCAM(model, model.get_submodule(gcfg["target_layer"]))
    to_input = eval_transform(config["images"])
    size = config["images"]["input_size"]
    to_photo = T.Compose([T.Resize(size), T.CenterCrop(size)])   # what the model sees, unnormalised

    cases = load_cv_pool(paths)
    oof = pd.read_csv(paths["predictions_csv"], dtype={"case_id": str})
    assert oof.case_id.is_unique and set(oof.case_id) == set(cases.case_id), \
        "OOF predictions must cover exactly the CV pool"
    oof = oof.merge(cases[["case_id", "efst_group"]], on="case_id", validate="one_to_one")
    oof["correct"] = oof.pred == oof.label
    oof["pred_prob"] = oof.apply(lambda r: r[f"prob_{r.pred}"], axis=1)
    images = case_images(cases).merge(oof[["case_id", "label", "pred", "efst_group", "correct"]],
                                      on="case_id")

    def heatmap(image_path, class_name):
        with Image.open(cache_path(paths["image_cache_dir"], image_path)) as im:
            im = im.convert("RGB")
            x = to_input(im).unsqueeze(0).to(device)
            cam, probs = cam_fn(x, classes.index(class_name))
            return cam, probs, np.asarray(to_photo(im))

    # 1. Every CV image: heatmap toward the class the CV model predicted
    frac = gcfg["border_fraction"]
    rows, sums, counts = [], {}, {}
    for r in tqdm(images.itertuples(), total=len(images), desc="Grad-CAM"):
        cam, probs, _ = heatmap(r.image_path, r.pred)
        rows.append({"case_id": r.case_id, "image_path": r.image_path, "efst_group": r.efst_group,
                     "label": r.label, "cv_pred": r.pred, "cv_correct": r.correct,
                     "final_pred": classes[int(probs.argmax())],
                     "border_share": border_share(cam, frac), "peak_in_border": peak_in_border(cam, frac)})
        if isinstance(r.efst_group, str):
            sums[r.efst_group] = sums.get(r.efst_group, 0) + cam
            counts[r.efst_group] = counts.get(r.efst_group, 0) + 1
    stats = pd.DataFrame(rows)
    Path(paths["gradcam_stats_csv"]).parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(paths["gradcam_stats_csv"], index=False)

    # 2. Border statistics, one value per case (mean over its images)
    per_case = stats.groupby("case_id").agg(efst_group=("efst_group", "first"),
                                            correct=("cv_correct", "first"),
                                            border_share=("border_share", "mean"),
                                            peak_in_border=("peak_in_border", "mean"))
    summary = {"all": mean_ci(per_case.border_share, ev["n_bootstrap"], ev["ci_level"], rng)}
    for group in EFST_ORDER:
        for correct, name in ((True, "correct"), (False, "incorrect")):
            sel = per_case[(per_case.efst_group == group) & (per_case.correct == correct)]
            if len(sel):
                summary[f"{group}, {name}"] = mean_ci(sel.border_share, ev["n_bootstrap"],
                                                      ev["ci_level"], rng)
    section = {
        "description": "Phase 6. Grad-CAM from the final baseline model (trained on the whole CV "
                       "pool) on every CV image, toward the class the seed-42 out-of-fold model "
                       "predicted. Correct / incorrect = out-of-fold. The test set was not used.",
        "config": args.config,
        "final_model": {k: info[k] for k in ("seed", "epochs", "n_cases", "n_images", "seconds",
                                             "gpu", "torch_version")},
        "target_layer": gcfg["target_layer"],
        "n_images": int(len(stats)),
        "border_fraction": frac,
        "border_share_if_uniform": round(uniform_border_share(size, frac), 4),
        "border_share_by_group": summary,
        "share_of_images_peaking_in_border": round(float(stats.peak_in_border.mean()), 4),
        "note": "border_share = share of the heatmap's mass in the outer border band. If the "
                "model looked everywhere equally it would equal border_share_if_uniform; much "
                "higher would suggest it relies on borders, backgrounds or framing.",
    }
    metrics_path = Path(paths["metrics_json"])
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["phase6_gradcam"] = section
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    plot_mean_cams({g: sums[g] / counts[g] for g in EFST_ORDER if g in counts}, counts, frac,
                   Path(paths["figures_dir"]) / "phase6_mean_gradcam.png")

    # 3. Illustration grids (first image of each selected case)
    picks, missing = select_cases(oof, classes, gcfg, rng)
    out_dir = Path(paths["gradcam_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    first_image = case_images(cases).groupby("case_id").image_path.first()
    shown = []
    for r in picks.itertuples():
        cam_pred, probs, photo = heatmap(first_image[r.case_id], r.pred)
        cam_true = cam_pred if r.correct else heatmap(first_image[r.case_id], r.label)[0]
        shown.append({"efst_group": r.efst_group, "label": r.label, "pred": r.pred,
                      "pred_prob": r.pred_prob, "correct": r.correct, "photo": photo,
                      "cam_pred": cam_pred, "cam_true": cam_true,
                      "border_pred": border_share(cam_pred, frac),
                      "final_pred": classes[int(probs.argmax())]})
    shown = pd.DataFrame(shown)
    for group in EFST_ORDER:
        for correct, name in ((True, "correct"), (False, "incorrect")):
            sel = shown[(shown.efst_group == group) & (shown.correct == correct)]
            miss = [m for m in missing if m["efst_group"] == group and m["correct"] == correct]
            plot_grid(sel, miss, f"{group}: {name} out-of-fold predictions (random picks; heatmaps "
                      "from the final model, which was trained on these cases)",
                      gcfg["overlay_alpha"], out_dir / f"gradcam_{slug(group)}_{name}.png")
    print(json.dumps({k: section[k] for k in ("border_share_if_uniform", "border_share_by_group",
                                               "share_of_images_peaking_in_border")}, indent=2))
    print(f"Wrote grids to {out_dir}, {Path(paths['figures_dir']) / 'phase6_mean_gradcam.png'} "
          f"and {metrics_path}")


if __name__ == "__main__":
    main()
