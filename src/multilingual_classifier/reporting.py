from pathlib import Path

import pandas as pd

from .utils import ensure_dir, write_json


def save_metrics(metrics: dict, output_dir: str | Path, filename: str = "metrics.json") -> None:
    write_json(metrics, Path(output_dir) / filename)


def write_markdown_report(
    overall: dict,
    language_metrics: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    ensure_dir(Path(output_path).parent)
    lines = [f"# {title}", "", "## Overall metrics", "", "| Metric | Value |", "|---|---:|"]
    for key in ("accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"):
        if key in overall:
            lines.append(f"| {key} | {overall[key]:.4f} |")
    lines.extend(["", "## Metrics by language", "", language_metrics.to_markdown(index=False), ""])
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
