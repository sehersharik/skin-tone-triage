"""Train the frozen method once on the whole CV pool and save its weights.

The frozen method is fixed in the config: architecture, image processing,
optimiser and epoch budget were all chosen on CV results in Phases 3-5. This
script only trains it on all CV cases (no fold held out) with the config's
seed, so the final model has seen as much data as possible. The test set is
dropped before training, as in src.train; it stays untouched for the one final
evaluation.

Weights go to `paths.final_model` (gitignored), with a small JSON describing
how the model was made next to it.

Usage:
    python -m src.train_final --config configs/final.yaml
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.data import case_images
from src.train import get_device, load_cv_pool, sync, train_model
from src.utils import load_config, save_metrics, set_seed, start_run


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths, tcfg = config["paths"], config["train"]
    if not isinstance(tcfg["epochs"], int):
        raise SystemExit("train.epochs must be a fixed number for the final model")
    set_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run_dir = start_run(paths["runs_dir"], f"train_final_{config['experiment']}", config)

    cases = load_cv_pool(paths)
    classes = sorted(cases.label.unique())
    cases["label_idx"] = cases.label.map(classes.index)
    images = case_images(cases).merge(cases[["case_id", "label", "label_idx"]], on="case_id")
    device = get_device(tcfg["device"])
    print(f"Final model: {len(cases)} cases, {len(images)} images, {tcfg['epochs']} epochs on {device}")

    sync(device); start = time.time()
    model = train_model(images, tcfg["epochs"], config, len(classes), device)
    sync(device)

    out = Path(paths["final_model"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    info = {
        "description": "Frozen method trained on the whole CV pool; the test set was not used.",
        "config": args.config, "seed": config["seed"], "epochs": tcfg["epochs"],
        "model": config["model"], "images": config["images"], "classes": classes,
        "n_cases": len(cases), "n_images": len(images),
        "seconds": round(time.time() - start, 1), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    Path(paths["final_model_info"]).write_text(json.dumps(info, indent=2) + "\n")
    save_metrics(run_dir, info)
    print(f"Wrote {out} and {paths['final_model_info']}. Run log: {run_dir}")


if __name__ == "__main__":
    main()
