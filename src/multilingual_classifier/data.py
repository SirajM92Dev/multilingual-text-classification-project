from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from .config import DatasetSpec


def _read_spec(spec: DatasetSpec) -> pd.DataFrame:
    if spec.source == "csv":
        if not spec.path:
            raise ValueError(f"CSV dataset '{spec.name}' requires path")
        frame = pd.read_csv(spec.path)
    elif spec.source == "huggingface":
        if not spec.dataset_name:
            raise ValueError(f"Hugging Face dataset '{spec.name}' requires dataset_name")
        loaded = load_dataset(spec.dataset_name, spec.dataset_config, split=spec.split)
        frame = loaded.to_pandas()
    else:
        raise ValueError(f"Unsupported dataset source: {spec.source}")

    missing = {spec.text_column, spec.label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset '{spec.name}' is missing columns: {sorted(missing)}")
    frame = frame[[spec.text_column, spec.label_column]].rename(
        columns={spec.text_column: "text", spec.label_column: "label"}
    )
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["label"] = frame["label"].astype(str)
    frame["language"] = spec.language
    frame["dataset"] = spec.name
    return frame[frame["text"].ne("")].reset_index(drop=True)


def load_all_data(specs: list[DatasetSpec]) -> pd.DataFrame:
    frames = [_read_spec(spec) for spec in specs]
    if not frames:
        raise ValueError("At least one dataset must be configured")
    data = pd.concat(frames, ignore_index=True)
    if data["label"].nunique() < 2:
        raise ValueError("At least two labels are required")
    return data


def split_data(
    data: pd.DataFrame,
    validation_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdout = validation_size + test_size
    if not 0 < holdout < 1:
        raise ValueError("validation_size + test_size must be between 0 and 1")
    train, temporary = train_test_split(
        data,
        test_size=holdout,
        random_state=seed,
        stratify=data["label"],
    )
    relative_test_size = test_size / holdout
    validation, test = train_test_split(
        temporary,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temporary["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def to_hf_dataset(frame: pd.DataFrame) -> Dataset:
    columns = ["text", "language", "dataset"]
    result = frame[columns].copy()
    result["labels"] = frame["label_id"].astype(int)
    return Dataset.from_pandas(result, preserve_index=False)


def to_dataset_dict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> DatasetDict:
    return DatasetDict(
        train=to_hf_dataset(train),
        validation=to_hf_dataset(validation),
        test=to_hf_dataset(test),
    )


def encode_labels(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = sorted(train["label"].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    frames = []
    for frame in (train, validation, test):
        unknown = set(frame["label"]) - set(label_to_id)
        if unknown:
            raise ValueError(f"Evaluation split contains labels absent from training: {sorted(unknown)}")
        copy = frame.copy()
        copy["label_id"] = copy["label"].map(label_to_id).astype(int)
        frames.append(copy)
    return label_to_id, frames[0], frames[1], frames[2]


def save_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / "train.csv", index=False)
    validation.to_csv(directory / "validation.csv", index=False)
    test.to_csv(directory / "test.csv", index=False)
