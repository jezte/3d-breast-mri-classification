
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration and resolve path entries relative to the project root."""
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    project_root = config_path.parent
    paths = config.get("paths", {})
    config["paths"] = {
        key: _resolve_path(value, project_root)
        for key, value in paths.items()
    }
    return config


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """Create directories required by configured outputs."""
    for key in ("extracted_root", "features_csv", "train_csv", "test_csv", "reports_dir"):
        path = config["paths"].get(key)
        if path is not None:
            directory = path if key in {"extracted_root", "reports_dir"} else path.parent
            directory.mkdir(parents=True, exist_ok=True)
