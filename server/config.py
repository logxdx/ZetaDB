"""Configuration loading. Single source of truth: config.toml at the repo root."""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    def __getitem__(self, key: str) -> Any:
        return self._d[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        """cfg.get('gpu.budget_mb') -> int"""
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the repo root, expanding ~."""
        raw = self.get(dotted)
        if raw is None:
            raise KeyError(dotted)
        p = Path(str(raw)).expanduser()
        return p if p.is_absolute() else (ROOT / p).resolve()

    def paths(self, dotted: str) -> list[Path]:
        out = []
        for raw in self.get(dotted, []) or []:
            p = Path(str(raw)).expanduser()
            out.append(p if p.is_absolute() else (ROOT / p).resolve())
        return out

    @property
    def models(self) -> dict[str, dict[str, Any]]:
        return self._d.get("models", {})


@lru_cache(maxsize=1)
def load(path: Path | None = None) -> Config:
    p = path or (ROOT / "config.toml")
    with p.open("rb") as fh:
        return Config(tomllib.load(fh))
