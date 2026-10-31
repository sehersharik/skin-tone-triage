"""Train the baseline with 5-fold grouped CV and save out-of-fold case predictions.

Only CV-pool cases are loaded; test cases are dropped right after the split
fingerprint check, so no code path here can train or predict on them.

1. Epoch budget: if `train.epochs` is null, train once per candidate budget on
   the folds other than `budget_fold`, score `budget_fold` (case-level balanced
   accuracy), and keep the best. The choice is recorded.
2. Cross-validation: for each fold, train on the other 4 folds for exactly that
   budget (cosine schedule, no early stopping) and predict the held-out fold.
   Image probabilities are averaged within each case.

Resuming: every budget candidate and every CV fold is saved to
`paths.fold_outputs_dir` as soon as it finishes, and a rerun skips the saved
ones. Each fold reseeds before training, so a resumed run gives the same
results as an uninterrupted one. The folder also holds the config it was made
with; a rerun with a different config stops instead of mixing results.

Several seeds: if the config has a `seeds` list, the whole CV run is repeated
once per seed on the same folds. `{seed}` and `{experiment}` in the paths are
filled in, and each seed's folds are saved in their own `seed<N>/` subfolder.
The epoch budget search (if any) runs once, with the first seed, and every
seed then uses the budget it picked.

Usage:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/baseline.yaml --fold-outputs-dir /kaggle/working/baseline_folds
    python -m src.train --config configs/seeds.yaml --fold-outputs-dir /kaggle/working/seed_folds
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data import case_images, split_fingerprint
from src.evaluate import case_metrics
from src.models import build_model
from src.prepare_images import cache_path
from src.transforms import eval_transform, train_transform
from src.utils import load_config, save_metrics, set_seed, start_run


class ImageDataset(Dataset):
    def __init__(self, images, cache_dir, transform):
        self.paths = [cache_path(cache_dir, p) for p in images.image_path]
        self.labels = images.label_idx.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB")), self.labels[i]


def load_cv_pool(paths):
    """Read cases.csv, check it matches the committed splits, return CV cases only."""
    cases = pd.read_csv(paths["cases_csv"], dtype={"case_id": str})
    expected = Path(paths["split_fingerprint"]).read_text().strip()
    if split_fingerprint(cases) != expected:
        raise SystemExit("cases.csv does not match reports/split_fingerprint.txt: "
                         "the splits differ from the committed ones")
    return cases[cases.split == "cv"].reset_index(drop=True)


def get_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(labels, n_classes):
    """Inverse-frequency weights: n / (n_classes * n_c), so a balanced set gives all 1s."""
    counts = np.bincount(labels, minlength=n_classes)
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


def make_loader(images, config, transform, shuffle, seed=0):
    return DataLoader(
        ImageDataset(images, config["paths"]["image_cache_dir"], transform),
        batch_size=config["train"]["batch_size"], shuffle=shuffle,
        num_workers=config["train"]["num_workers"],
        generator=torch.Generator().manual_seed(seed), pin_memory=torch.cuda.is_available())


def train_model(images, epochs, config, n_classes, device):
    set_seed(config["seed"])
    tcfg = config["train"]
    loader = make_loader(images, config, train_transform(config["images"]), shuffle=True,
                         seed=config["seed"])
    model = build_model(config["model"]["name"], n_classes, config["model"]["pretrained"]).to(device)
    weight = class_weights(images.label_idx.to_numpy(), n_classes).to(device) \
        if tcfg["class_weighted_loss"] else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(y)
        print(f"    epoch {epoch + 1}/{epochs}  train loss {total / len(images):.4f}", flush=True)
    return model


@torch.no_grad()
def predict_cases(model, images, config, classes, device):
    """Softmax per image, then the mean over each case's images."""
    model.eval()
    loader = make_loader(images, config, eval_transform(config["images"]), shuffle=False)
    probs = torch.cat([torch.softmax(model(x.to(device)), dim=1).cpu() for x, _ in loader])
    image_probs = pd.DataFrame(probs.numpy(), columns=[f"prob_{c}" for c in classes])
    image_probs["case_id"] = images.case_id.to_numpy()
    return image_probs.groupby("case_id", sort=True).mean().reset_index()


class FoldStore:
    """Saves each finished budget candidate / CV fold to disk and reloads it on a rerun."""

    def __init__(self, folder, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        saved = self.folder / "config.yaml"
        if saved.exists():
            if yaml.safe_load(saved.read_text()) != config:
                raise SystemExit(f"{self.folder} holds results from a different config: "
                                 "remove it or choose another --fold-outputs-dir")
        else:
            saved.write_text(yaml.safe_dump(config, sort_keys=False))

    def _write(self, name, text):
        tmp = self.folder / f"{name}.tmp"
        tmp.write_text(text)
        tmp.replace(self.folder / name)  # atomic: a timeout mid-write leaves no half file

    def load(self, name):
        info = self.folder / f"{name}.json"
        if not info.exists():
            return None, None
        csv = self.folder / f"{name}.csv"
        preds = pd.read_csv(csv, dtype={"case_id": str}) if csv.exists() else None
        return json.loads(info.read_text()), preds

    def save(self, name, info, preds=None):
        if preds is not None:  # predictions first, so a saved .json always has its .csv
            self._write(f"{name}.csv", preds.to_csv(index=False))
        self._write(f"{name}.json", json.dumps(info, indent=2) + "\n")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def seed_config(config, seed):
    """The config for one seed of a multi-seed run: `seed` set, `{seed}` filled in the paths."""
    config = copy.deepcopy(config)
    del config["seeds"]
    config["seed"] = seed
    config["paths"] = {k: v.format(seed=seed, experiment=config["experiment"])
                       for k, v in config["paths"].items()}
    return config


def run_cv(config, fold_outputs_dir, budget_log=None):
    """Epoch budget (if not fixed or given) + 5-fold CV for one seed; writes predictions and
    the train log. `budget_log` reuses a budget already searched with another seed.
    Returns the budget log."""
    paths, tcfg = config["paths"], config["train"]
    store = FoldStore(fold_outputs_dir, config)
    set_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run_dir = start_run(paths["runs_dir"], f"train_{config['experiment']}_seed{config['seed']}",
                        config)

    cases = load_cv_pool(paths)
    classes = sorted(cases.label.unique())
    cases["label_idx"] = cases.label.map(classes.index)
    images = case_images(cases).merge(cases[["case_id", "label", "label_idx", "cv_fold"]],
                                      on="case_id")
    device = get_device(tcfg["device"])
    print(f"Device: {device}  cases: {len(cases)}  images: {len(images)}  classes: {classes}")

    def run_fold(val_fold, epochs):
        model = train_model(images[images.cv_fold != val_fold], epochs, config, len(classes), device)
        preds = predict_cases(model, images[images.cv_fold == val_fold], config, classes, device)
        return preds.merge(cases[["case_id", "label"]], on="case_id")

    # 1. Epoch budget, chosen on one fold only
    if budget_log is None:
        budget_log = {"selected": tcfg["epochs"], "source": "config"}
    epochs = budget_log["selected"]
    if epochs is None:
        fold, scores, budget_seconds = tcfg["budget_fold"], {}, {}
        for candidate in tcfg["candidate_epochs"]:
            saved, _ = store.load(f"budget_{candidate}")
            if saved is None:
                print(f"Budget search: {candidate} epochs, scored on cv_fold {fold}")
                sync(device); t0 = time.time()
                preds = run_fold(fold, candidate)
                sync(device)
                probs = preds[[f"prob_{c}" for c in classes]].to_numpy()
                saved = {"balanced_accuracy": round(
                             case_metrics(preds.label, probs, classes)["balanced_accuracy"], 4),
                         "seconds": round(time.time() - t0, 1)}
                store.save(f"budget_{candidate}", saved)
            else:
                print(f"Budget search: {candidate} epochs already done, loaded from disk")
            scores[candidate] = saved["balanced_accuracy"]
            budget_seconds[candidate] = saved["seconds"]
            print(f"  balanced accuracy: {scores[candidate]}")
        epochs = max(scores, key=lambda e: (scores[e], -e))  # ties -> fewer epochs
        budget_log = {"selected": epochs, "source": "budget search", "seed": config["seed"],
                      "budget_fold": fold,
                      "metric": "case-level balanced accuracy", "scores": scores,
                      "seconds_per_candidate": budget_seconds}
        print(f"Selected epoch budget: {epochs}")

    # 2. Cross-validation with the fixed budget
    oof, fold_seconds = [], {}
    for fold in sorted(cases.cv_fold.unique()):
        name = f"cv_fold_{fold}_{epochs}ep"
        saved, preds = store.load(name)
        if saved is None:
            print(f"CV fold {fold}: {epochs} epochs")
            sync(device); start = time.time()
            preds = run_fold(fold, epochs)
            sync(device)
            preds["cv_fold"] = fold
            saved = {"seconds": round(time.time() - start, 1)}
            store.save(name, saved, preds)
        else:
            print(f"CV fold {fold}: already done, loaded from disk")
        fold_seconds[int(fold)] = saved["seconds"]
        oof.append(preds)
        print(f"  fold {fold} took {fold_seconds[int(fold)]} s")

    oof = pd.concat(oof, ignore_index=True)
    oof["pred"] = np.asarray(classes)[oof[[f"prob_{c}" for c in classes]].to_numpy().argmax(axis=1)]
    Path(paths["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(paths["predictions_csv"], index=False)

    log = {
        "seed": config["seed"],
        "classes": classes,
        "n_cases": len(cases), "n_images": len(images),
        "epoch_budget": budget_log,
        "timing": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "seconds_per_cv_fold": fold_seconds,
            "mean_seconds_per_cv_fold": round(float(np.mean(list(fold_seconds.values()))), 1),
        },
        "torch_version": torch.__version__,
    }
    Path(paths["train_log_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["train_log_json"]).write_text(json.dumps(log, indent=2) + "\n")
    save_metrics(run_dir, log)
    oof.to_csv(run_dir / "oof_predictions.csv", index=False)
    print(f"Wrote {paths['predictions_csv']} and {paths['train_log_json']}. Run log: {run_dir}")
    return budget_log


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold-outputs-dir", help="overrides paths.fold_outputs_dir "
                        "(e.g. a folder that survives a Kaggle timeout)")
    args = parser.parse_args()
    config = load_config(args.config)
    if "seeds" not in config:
        run_cv(config, args.fold_outputs_dir or config["paths"]["fold_outputs_dir"])
        return
    budget_log = None  # searched with the first seed, then reused
    for seed in config["seeds"]:
        print(f"===== {config['experiment']}: seed {seed} =====")
        cfg = seed_config(config, seed)
        folder = Path(args.fold_outputs_dir) / f"seed{seed}" if args.fold_outputs_dir \
            else cfg["paths"]["fold_outputs_dir"]
        budget_log = run_cv(cfg, folder, budget_log)


if __name__ == "__main__":
    main()
