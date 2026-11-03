import argparse
from pathlib import Path

from sklearn.preprocessing import LabelEncoder

from multilingual_classifier.config import dataset_specs, load_config
from multilingual_classifier.data import encode_labels, load_all_data, save_splits, split_data
from multilingual_classifier.metrics import classification_metrics, metrics_by_language
from multilingual_classifier.models import TfidfBaseline
from multilingual_classifier.reporting import save_metrics
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
    output = ensure_dir(Path(config["output_dir"]) / "baseline")
    save_splits(train, validation, test, output / "data")
    model = TfidfBaseline().fit(train["text"], train["label_id"])
    model.save(output)
    predictions = model.predict(test["text"])
    result = test.assign(prediction=predictions)
    overall = classification_metrics(result["label_id"], result["prediction"])
    language = metrics_by_language(result)
    save_metrics(
        {"label_to_id": label_to_id, "overall": overall, "by_language": language.to_dict("records")},
        output,
    )
    language.to_csv(output / "language_metrics.csv", index=False)
    write_json(label_to_id, output / "label_map.json")


if __name__ == "__main__":
    main()
