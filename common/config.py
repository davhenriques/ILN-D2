"""Configuration loading shared by every module.

`load_config()` reads config.yaml (or the file in $D2_CONFIG) and resolves every
path in `paths` to an absolute path under the project root.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("D2_CONFIG", ROOT / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: str((ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    return cfg


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    # Silence chatty third-party loggers
    for name in ("httpx", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
