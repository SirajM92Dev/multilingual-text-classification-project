import argparse
from pathlib import Path

from multilingual_classifier.config import dataset_specs, load_config
from multilingual_classifier.data import (
    encode_labels,
    load_all_data,
    save_splits,
    split_data,
    to_dataset_dict,
)
from multilingual_classifier.metrics import classification_metrics, metrics_by_language
from multilingual_classifier.models import train_transformer
from multilingual_classifier.reporting import save_metrics, write_markdown_report
from multilingual_classifier.utils import ensure_dir, set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    train, validation, test = split_data(
        load_all_data(dataset_specs(config)),
        config["split"]["validation_size"],
        config["split"]["test_size"],
        config["seed"],
    )
    label_to_id, train, validation, test = encode_labels(train, validation, test)
    output = ensure_dir(Path(config["output_dir"]) / "transformer")
    save_splits(train, validation, test, output / "data")
    datasets = to_dataset_dict(train, validation, test)
    model_config = config["model"]
    train_transformer(
        datasets=datasets,
        model_name=model_config["name"],
        output_dir=output,
        num_labels=len(label_to_id),
        max_length=model_config["max_length"],
        epochs=model_config["num_train_epochs"],
        learning_rate=model_config["learning_rate"],
        train_batch_size=model_config["train_batch_size"],
        eval_batch_size=model_config["eval_batch_size"],
        weight_decay=model_config["weight_decay"],
        warmup_ratio=model_config["warmup_ratio"],
        gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
        fp16=model_config["fp16"],
    )
    write_json(label_to_id, output / "label_map.json")
    write_json({"model": model_config["name"], "max_length": model_config["max_length"]}, output / "model_config.json")


if __name__ == "__main__":
    main()
