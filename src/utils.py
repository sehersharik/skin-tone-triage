"""Small helpers shared by every script: config loading, seeding, run logging."""
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def merge(base, override):
    """Recursively merge two dicts; values in `override` win."""
    out = dict(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(
            out.get(key), dict) else value
    return out


def load_config(path):
    """Load a YAML config. If it has `base: <path>`, start from that config and
    apply this file's values on top (so experiment configs list only what differs)."""
    with open(path) as f:
        config = yaml.safe_load(f)
    if "base" in config:
        config = merge(load_config(config.pop("base")), config)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def start_run(runs_dir, name, config):
    """Create runs/<timestamp>_<name>/ and save the config used for this run."""
    run_dir = Path(runs_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return run_dir


def save_metrics(run_dir, metrics):
    with open(Path(run_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
