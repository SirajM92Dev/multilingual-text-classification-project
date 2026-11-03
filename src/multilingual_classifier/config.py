from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    text_column: str
    label_column: str
    language: str
    dataset_name: str | None = None
    dataset_config: str | None = None
    path: str | None = None
    split: str = "train"


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dataset_specs(config: dict[str, Any]) -> list[DatasetSpec]:
    return [DatasetSpec(**item) for item in config["datasets"]]
