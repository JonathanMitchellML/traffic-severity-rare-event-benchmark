"""Shared utilities for configuration, paths, and column normalization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def to_snake_case(name: object) -> str:
    """Convert NYC Open Data style column names to stable snake_case names."""
    text = str(name).strip()
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", "_", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def normalize_column_names(df):
    """Return a copy of a dataframe with de-duplicated snake_case columns."""
    renamed = []
    seen: dict[str, int] = {}
    for column in df.columns:
        base_name = to_snake_case(column)
        count = seen.get(base_name, 0)
        seen[base_name] = count + 1
        renamed.append(base_name if count == 0 else f"{base_name}_{count + 1}")

    out = df.copy()
    out.columns = renamed
    return out


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    resolved = resolve_project_path(config_path)
    with resolved.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory path."""
    resolved = resolve_project_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def config_path(config: dict[str, Any], *keys: str, default: str) -> Path:
    """Read a nested path from config and resolve it against the project root."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return resolve_project_path(default)
        value = value[key]
    return resolve_project_path(value)

