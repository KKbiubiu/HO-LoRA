# Evaluation protocol

## Included evidence

Six evaluation JSON files preserve the recorded Standard LoRA baseline, local B-LoRA baseline, HO-LoRA v10, HO-LoRA v12, no-orthogonal-loss ablation and no-Texture-Solo ablation. Their [provenance records](../results/provenance.json) contain SHA-256 hashes. Each source experiment contains 120 generated images: 15 prompts × eight variants.

The historical inference defaults are seed 42, 25 steps, guidance scale 7.5 and a DPM-Solver multistep scheduler. The [prompt list](../results/prompts.json) includes the negative prompt and these defaults. Defaults are not per-image execution receipts: the original JSON does not retain the full environment, exact model revision or independent-run metadata.

## References and metrics

Each metric compares a variant with the `full` output for the same prompt **within that experiment**. The reference changes when the trained model changes.

| Metric | Definition | Interpretation |
| :--- | :--- | :--- |
| ColorHist | OpenCV HSV histogram χ² distance; 16 bins per channel, normalized histogram | Lower means closer color statistics to that experiment's full output |
| EdgeIoU | Canny edge intersection over union, thresholds 50/150 | Higher means greater edge overlap with that full output |
| GramDist | Mean Frobenius distance of normalized VGG-19 Gram matrices | Lower means closer VGG feature statistics to that full output |
| ΔT | GramDist(no_lora, full) − GramDist(texture_only, full) | Positive means Texture-only improves over the base model relative to that full reference |

The VGG feature slices end at indices 4, 9, 18 and 27, with resize to 256, center crop to 224 and ImageNet normalization. Those slice indices define the implementation more precisely than informal layer names.

The JSON `std` fields describe variation across evaluated prompts. They do not measure uncertainty across independent training runs. Paired per-prompt metrics were not retained, so the records cannot reconstruct a confidence interval for ΔT.

## Baseline interpretation

The baseline values come from this project's local implementations. The local B-LoRA configuration has Content and Style branches; its `texture_only` label reuses Style. It has no independently trained Texture branch. A larger B-LoRA ΔT therefore does not establish finer texture–color separation.

## Known limitations

- CLIP-I is absent from all six released JSON files. The historical evaluation records include a CLIP weight-download failure; absent scores must not be treated as zero.
- The original evaluator could fall back to random VGG-19 weights when pretrained weights were unavailable. The JSON files do not encode VGG weight provenance. A new evaluation needs verified pretrained weights and must fail if they cannot be loaded.
- The orthogonality diagnostic concatenates and truncates vectors of unequal size. It differs from the cross-Gram training loss and does not prove semantic independence.
- The saved training coefficient and retained historical implementation disagree; see [the method notes](METHOD.md).
- Evidence covers one artist, one SDXL backbone and manually chosen branch locations. Independent training repeats, broader datasets and human evaluation are not established by these records.

## Reproduce the summaries

Run `python3 tools/summarize_results.py --check` to verify the tables, or omit `--check` to regenerate them. This uses the recorded metric means and reproduces their arithmetic; it does not recompute metrics from images.

The result differences support a descriptive comparison in this setting. They do not support percentage causal attribution, statistical significance, a universal quality ranking or independent control of every visual attribute.

[Complete results](../results/SUMMARY.md) · [Project overview](../README.md)
