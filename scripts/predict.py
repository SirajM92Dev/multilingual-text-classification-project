import argparse
import json
from pathlib import Path

from multilingual_classifier.models import predict_transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/transformer")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    prediction = int(predict_transformer(args.model_dir, [args.text])[0])
    label_map = json.loads((Path(args.model_dir) / "label_map.json").read_text(encoding="utf-8"))
    inverse = {int(value): key for key, value in label_map.items()}
    print(json.dumps({"text": args.text, "label_id": prediction, "label": inverse[prediction]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
