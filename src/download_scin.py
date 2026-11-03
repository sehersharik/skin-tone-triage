"""Download the SCIN dataset from its public Google Cloud Storage bucket.

Follows the official access method in scin_demo.ipynb (google-cloud-storage
client, bucket `dx-scin-public-data`). The bucket is public, so an anonymous
client is used and no Google login is needed.

Files are mirrored under paths.raw_dir with the same relative paths as in the
bucket (e.g. data/raw/scin/dataset/images/<id>.png), so the image paths stored
in scin_cases.csv can be joined onto raw_dir directly.

Usage:
    python -m src.download_scin --config configs/data.yaml
    python -m src.download_scin --config configs/data.yaml --images used   # after build_metadata
    python -m src.download_scin --config configs/data.yaml --images test   # test-split images only
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from google.cloud import storage
from tqdm import tqdm

from src.data import case_images
from src.utils import load_config, save_metrics, set_seed, start_run


def download_blob(bucket, blob_name, dest):
    """Download one file unless an identical-size copy already exists."""
    blob = bucket.blob(blob_name)
    blob.reload()  # fetch size so we can skip finished files
    if dest.exists() and dest.stat().st_size == blob.size:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    blob.download_to_filename(tmp)
    tmp.rename(dest)
    return "downloaded"


def image_paths_from_cases(cases_csv, path_columns):
    cases = pd.read_csv(cases_csv, dtype={"case_id": str})
    paths = pd.concat([cases[c] for c in path_columns]).dropna()
    return sorted(paths.unique())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--images", choices=["all", "used", "test", "sample", "none"],
                        help="override download.images from the config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.images:
        config["download"]["images"] = args.images
    set_seed(config["seed"])

    scin, dl = config["scin"], config["download"]
    raw_dir = Path(config["paths"]["raw_dir"])
    run_dir = start_run(config["paths"]["runs_dir"], "download_scin", config)

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(scin["bucket"])

    # 1. Metadata CSVs
    for name in scin["csv_files"]:
        status = download_blob(bucket, name, raw_dir / name)
        print(f"{name}: {status}")

    # 2. Images (the list comes from the cases CSV, not from guessing names)
    cases_csv = raw_dir / "dataset/scin_cases.csv"
    all_paths = image_paths_from_cases(cases_csv, scin["image_path_columns"])
    if dl["images"] == "all":
        wanted = all_paths
    elif dl["images"] in ("used", "test"):   # test = only the test split's images
        cases_csv_used = Path(config["paths"]["cases_csv"])
        if not cases_csv_used.exists():
            raise SystemExit(f"{cases_csv_used} not found: run src.build_metadata first")
        cases = pd.read_csv(cases_csv_used, dtype={"case_id": str})
        if dl["images"] == "test":
            cases = cases[cases.split == "test"]
        wanted = sorted(case_images(cases).image_path.unique())
    elif dl["images"] == "sample":
        wanted = sorted(pd.Series(all_paths)
                        .sample(n=min(dl["sample_size"], len(all_paths)),
                                random_state=config["seed"]))
    else:
        wanted = []

    def fetch(path):
        try:
            return path, download_blob(bucket, path, raw_dir / path)
        except Exception as e:  # e.g. the known missing image (SCIN issue #1)
            return path, f"failed: {e.__class__.__name__}"

    results = {}
    with ThreadPoolExecutor(max_workers=dl["num_workers"]) as pool:
        for path, status in tqdm(pool.map(fetch, wanted), total=len(wanted),
                                 desc="images"):
            results[path] = status

    failed = sorted(p for p, s in results.items() if s.startswith("failed"))
    summary = {
        "image_mode": dl["images"],
        "images_referenced_in_cases_csv": len(all_paths),
        "images_requested": len(wanted),
        "downloaded": sum(s == "downloaded" for s in results.values()),
        "skipped_existing": sum(s == "skipped" for s in results.values()),
        "failed": failed,
    }
    save_metrics(run_dir, summary)
    print({k: v for k, v in summary.items() if k != "failed"},
          f"failed={len(failed)}")
    print(f"Run log: {run_dir}")


if __name__ == "__main__":
    main()
