"""Resize the images of the kept cases once and cache them.

SCIN photos are ~810x1080 PNGs. Decoding them every epoch is slow on Kaggle's
few CPU cores, so each image is resized once so its short side is
`images.cache_short_side` pixels (aspect ratio kept) and saved as PNG.
The training transforms then work on these small files.

Usage:
    python -m src.prepare_images --config configs/baseline.yaml
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data import case_images
from src.utils import load_config


def resize_short_side(image, short_side):
    w, h = image.size
    scale = short_side / min(w, h)
    return image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)


def cache_path(cache_dir, image_path):
    return Path(cache_dir) / Path(image_path).name


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    paths, short_side = config["paths"], config["images"]["cache_short_side"]

    cases = pd.read_csv(paths["cases_csv"], dtype={"case_id": str})
    image_paths = case_images(cases).image_path.unique()
    Path(paths["image_cache_dir"]).mkdir(parents=True, exist_ok=True)

    def process(image_path):
        src = Path(paths["raw_dir"]) / image_path
        dest = cache_path(paths["image_cache_dir"], image_path)
        if dest.exists():
            return "skipped"
        if not src.exists():
            return "missing"
        with Image.open(src) as im:
            resize_short_side(im.convert("RGB"), short_side).save(dest)
        return "resized"

    with ThreadPoolExecutor(max_workers=8) as pool:
        status = list(tqdm(pool.map(process, image_paths), total=len(image_paths), desc="resize"))
    counts = pd.Series(status).value_counts().to_dict()
    print(counts)
    if counts.get("missing"):
        raise SystemExit("Some raw images are missing: run src.download_scin --images used")


if __name__ == "__main__":
    main()
