import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import Trainer

from multilingual_classifier.config import load_config
from multilingual_classifier.data import to_dataset_dict
from multilingual_classifier.metrics import classification_metrics, metrics_by_language
from multilingual_classifier.models import tokenize_datasets
from multilingual_classifier.reporting import save_metrics, write_markdown_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default="artifacts/transformer")
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    test = pd.read_csv(model_dir / "data" / "test.csv")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dataset = to_dataset_dict(test, test, test)
    tokenized = tokenize_datasets(dataset, tokenizer, load_config(args.config)["model"]["max_length"])
    trainer = Trainer(model=model, tokenizer=tokenizer)
    output = trainer.predict(tokenized["test"])
    predictions = output.predictions.argmax(axis=-1)
    result = test.assign(prediction=predictions)
    overall = classification_metrics(result["label_id"], result["prediction"])
    language = metrics_by_language(result)
    save_metrics({"overall": overall, "by_language": language.to_dict("records")}, model_dir)
    language.to_csv(model_dir / "language_metrics.csv", index=False)
    write_markdown_report(overall, language, model_dir / "report.md", "Transformer Evaluation Report")


if __name__ == "__main__":
    main()
