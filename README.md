# Multilingual Text Classification with Transformer Models

This project compares a traditional TF-IDF + Logistic Regression baseline with a fine-tuned multilingual transformer for English and Urdu/Roman Urdu text classification.

It includes:

- Configurable Hugging Face or local CSV datasets.
- A reproducible train/validation/test split.
- A TF-IDF + Logistic Regression baseline.
- XLM-RoBERTa fine-tuning with PyTorch and Hugging Face Transformers.
- Separate English, Urdu, and Roman Urdu metrics.
- Code-switching and spelling-variation robustness evaluation.
- JSON, CSV, and Markdown reports.

## Recommended datasets

The default configuration uses binary sentiment datasets:

- **English:** `stanfordnlp/imdb`, binary movie-review sentiment.
- **Urdu:** `urduhack/urdu_sentiment_corpus`, binary sentiment classification.

For a strict same-label-space comparison, ensure the Urdu dataset uses the same positive/negative label semantics as the English dataset. If the Urdu source uses different label names, normalize them in the source CSV or add a label-normalization step before training.

The loader supports any Hugging Face dataset or CSV file with configurable `text_column`, `label_column`, and optional `language` value.

## Project layout

```text
multilingual-text-classification/
├── config.yaml
├── requirements.txt
├── README.md
├── data/
│   └── README.md
├── scripts/
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   ├── run_robustness.py
│   └── predict.py
└── src/
    └── multilingual_classifier/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── metrics.py
        ├── models.py
        ├── perturbations.py
        ├── reporting.py
        └── utils.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` before training. Each dataset entry can use either:

```yaml
source: huggingface
dataset_name: your-org/your-dataset
dataset_config: default
text_column: text
label_column: label
language: urdu
```

or:

```yaml
source: csv
path: data/urdu.csv
text_column: text
label_column: label
language: urdu
```

Labels are normalized to strings and encoded globally so English and Urdu use the same class IDs.

## Training

```bash
export PYTHONPATH=src
python scripts/train_baseline.py --config config.yaml
python scripts/train_transformer.py --config config.yaml
```

## Evaluation

```bash
export PYTHONPATH=src
python scripts/evaluate.py --config config.yaml --model-dir artifacts/transformer
python scripts/run_robustness.py --config config.yaml --model-dir artifacts/transformer
```

Outputs are written to `artifacts/`, including:

- `baseline/metrics.json`
- `transformer/metrics.json`
- `transformer/language_metrics.csv`
- `transformer/report.md`
- `robustness/robustness_metrics.csv`

## Prediction

```bash
export PYTHONPATH=src
python scripts/predict.py \
  --model-dir artifacts/transformer \
  --text "یہ فلم بہت اچھی ہے"
```

## Research questions answered

1. How does a TF-IDF baseline compare with multilingual XLM-R?
2. Does performance differ between English, Urdu, and Roman Urdu?
3. How much does code-switching reduce performance?
4. How sensitive is the model to common spelling variations?

Use the generated per-language metrics and robustness table in the project report.
