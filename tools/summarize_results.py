#!/usr/bin/env python3
"""Regenerate the published summary using only the Python standard library."""

import argparse
import csv
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    ("baseline_standard", "Standard LoRA (local baseline)"),
    ("baseline_blora", "B-LoRA (local baseline)*"),
    ("v10", "HO-LoRA v10"),
    ("ablation_3step", "HO-LoRA, without Texture-Solo"),
    ("ablation_no_ortho", "HO-LoRA, without orthogonal loss"),
    ("v12", "HO-LoRA v12"),
)
FIELDS = ("experiment", "method", "content_color_hist", "content_edge_iou", "texture_gram", "no_lora_gram", "texture_delta")


def summarize(data_dir):
    rows = []
    for experiment, method in EXPERIMENTS:
        data = json.loads((data_dir / f"eval_{experiment}.json").read_text(encoding="utf-8"))["per_variant"]

        def mean(variant, metric):
            value = data[variant][metric]["mean"]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Invalid metric: {experiment}/{variant}/{metric}")
            return value

        texture_gram = mean("texture_only", "gram_dist")
        no_lora_gram = mean("no_lora", "gram_dist")
        rows.append(dict(zip(FIELDS, (
            experiment, method,
            mean("content_only", "color_hist_dist"),
            mean("content_only", "edge_iou"),
            texture_gram, no_lora_gram, no_lora_gram - texture_gram,
        ))))
    return rows


def render(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    table = [
        "# Recorded experiment summary", "",
        "Generated from `raw/eval_*.json` by `tools/summarize_results.py`; no model inference is performed.", "",
        "| Method | Content ColorHist ↓ | Content EdgeIoU ↑ | Texture GramDist ↓ | ΔT ↑ |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(f"| {row['method']} | {row['content_color_hist']:.3f} | {row['content_edge_iou']:.3f} | {row['texture_gram']:.3f} | {row['texture_delta']:+.3f} |")
    table.extend([
        "", "ΔT = GramDist(no_lora, full) − GramDist(texture_only, full). Each experiment uses its own `full` reference.", "",
        "*The local B-LoRA baseline reuses the style branch for the texture label; it does not have a separately trained Texture branch.", "",
        "These are descriptive proxy metrics, not an independent image-quality leaderboard. See [evaluation protocol](../docs/EVALUATION.md).", "",
    ])
    return {"summary.csv": stream.getvalue(), "SUMMARY.md": "\n".join(table)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if committed summaries differ from the raw JSON.")
    args = parser.parse_args()
    outputs = render(summarize(ROOT / "results/raw"))
    for name, expected in outputs.items():
        path = ROOT / "results" / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"Summary differs: {path.name}. Run python tools/summarize_results.py")
        else:
            path.write_text(expected, encoding="utf-8")
    print("Result summaries verified." if args.check else "Result summaries written.")


if __name__ == "__main__":
    main()
