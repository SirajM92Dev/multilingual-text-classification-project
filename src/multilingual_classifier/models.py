from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class TfidfBaseline:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=200_000,
        )
        self.classifier = LogisticRegression(max_iter=1_000, class_weight="balanced")

    def fit(self, texts: pd.Series, labels: pd.Series) -> "TfidfBaseline":
        features = self.vectorizer.fit_transform(texts)
        self.classifier.fit(features, labels)
        return self

    def predict(self, texts: pd.Series) -> np.ndarray:
        return self.classifier.predict(self.vectorizer.transform(texts))

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        dump(self, path / "model.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "TfidfBaseline":
        return load(Path(directory) / "model.joblib")


def tokenize_datasets(dataset_dict: Any, tokenizer: Any, max_length: int) -> Any:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset_dict.map(tokenize, batched=True)


def train_transformer(
    datasets: Any,
    model_name: str,
    output_dir: str | Path,
    num_labels: int,
    max_length: int,
    epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    fp16: bool,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_datasets(datasets, tokenizer, max_length)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16 and torch.cuda.is_available(),
        report_to="none",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer, tokenized


def predict_transformer(
    model_dir: str | Path,
    texts: list[str],
    max_length: int = 160,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    return logits.argmax(dim=-1).cpu().numpy()
