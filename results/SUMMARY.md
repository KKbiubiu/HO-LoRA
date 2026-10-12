# Recorded experiment summary

Generated from `raw/eval_*.json` by `tools/summarize_results.py`; no model inference is performed.

| Method | Content ColorHist ↓ | Content EdgeIoU ↑ | Texture GramDist ↓ | ΔT ↑ |
| :--- | ---: | ---: | ---: | ---: |
| Standard LoRA (local baseline) | 470.042 | 0.101 | 3.804 | +0.764 |
| B-LoRA (local baseline)* | 319.157 | 0.101 | 2.365 | +2.672 |
| HO-LoRA v10 | 224.273 | 0.112 | 4.049 | -0.018 |
| HO-LoRA, without Texture-Solo | 313.407 | 0.111 | 4.078 | -0.051 |
| HO-LoRA, without orthogonal loss | 49.270 | 0.127 | 3.229 | +1.091 |
| HO-LoRA v12 | 45.003 | 0.126 | 2.988 | +1.111 |

ΔT = GramDist(no_lora, full) − GramDist(texture_only, full). Each experiment uses its own `full` reference.

*The local B-LoRA baseline reuses the style branch for the texture label; it does not have a separately trained Texture branch.

These are descriptive proxy metrics, not an independent image-quality leaderboard. See [evaluation protocol](../docs/EVALUATION.md).
