import argparse
from pathlib import Path

import pandas as pd

from multilingual_classifier.metrics import metrics_by_language
from multilingual_classifier.models import predict_transformer
from multilingual_classifier.perturbations import build_perturbed, code_switch, spelling_variation
from multilingual_classifier.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-dir", default="artifacts/transformer")
    args = parser.parse_args()
    del args.config
    model_dir = Path(args.model_dir)
    test = pd.read_csv(model_dir / "data" / "test.csv")
    max_length = 160
    records = []
    for name, transform in (("code_switching", code_switch), ("spelling_variation", spelling_variation)):
        perturbed = build_perturbed(test, transform, name)
        perturbed["prediction"] = predict_transformer(model_dir, perturbed["text"].tolist(), max_length).tolist()
        metrics = metrics_by_language(perturbed)
        metrics["perturbation"] = name
        records.append(metrics)
    result = pd.concat(records, ignore_index=True)
    result.to_csv(model_dir.parent / "robustness_metrics.csv", index=False)
    write_json(result.to_dict("records"), model_dir.parent / "robustness_metrics.json")


if __name__ == "__main__":
    main()
