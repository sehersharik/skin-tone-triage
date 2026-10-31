"""Measure simple photo-quality factors for every image of the CV pool.

Per image:
- width, height, megapixels: of the original SCIN photo (read from the file header)
- brightness: mean grey level (0-255) of the cached 256 px image the model is trained on
- contrast: standard deviation of the grey levels (RMS contrast)
- sharpness: variance of the Laplacian of the grey image (low = blurry)

Brightness and contrast are measured on the whole photo, so they partly reflect
the skin itself: darker skin gives a darker photo. Sharpness and resolution do
not depend on skin tone in that direct way.

Only CV-pool images are measured; the test set is never opened.

Usage:
    python -m src.image_factors --config configs/seeds.yaml
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data import case_images
from src.prepare_images import cache_path
from src.train import load_cv_pool
from src.utils import load_config


def grey_stats(grey):
    """Brightness, contrast and sharpness of one greyscale uint8 image."""
    return {
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "sharpness": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
    }


def measure(raw_path, cached_path):
    with Image.open(raw_path) as im:  # opening reads only the header, not the pixels
        width, height = im.size
    grey = cv2.imread(str(cached_path), cv2.IMREAD_GRAYSCALE)
    if grey is None:
        raise SystemExit(f"Could not read {cached_path}: run src.prepare_images first")
    return {"width": width, "height": height, "megapixels": width * height / 1e6,
            **grey_stats(grey)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    paths = load_config(args.config)["paths"]

    images = case_images(load_cv_pool(paths))
    jobs = [(Path(paths["raw_dir"]) / p, cache_path(paths["image_cache_dir"], p))
            for p in images.image_path]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(tqdm(pool.map(lambda job: measure(*job), jobs), total=len(jobs),
                         desc="image factors"))
    factors = pd.concat([images, pd.DataFrame(rows)], axis=1)
    out = Path(paths["image_factors_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    factors.to_csv(out, index=False)
    print(factors[["megapixels", "brightness", "contrast", "sharpness"]].describe().round(2))
    print(f"Wrote {out} ({len(factors)} images)")


if __name__ == "__main__":
    main()
