# Release scope

HO-LoRA's public repository contains the core method implementation and selected visual results. It includes a callable training step, image evaluation functions and the standard-library utility for generating and verifying recorded summary tables.

## Included

- English and Chinese project overviews.
- Twelve original-resolution final results, a preview gallery, branch comparisons and source hashes.
- Core branch-specific LoRA, four-phase training and cross-Gram regularization.
- Core color, edge and pretrained VGG19 Gram metrics with paired evaluation and ΔT.
- Six recorded evaluation JSON files and generated summary tables.
- Evaluation prompts, recorded v12 settings and provenance.
- Method and evaluation documentation.
- Pinned core dependencies, API examples, focused tests and continuous-integration checks.

## Availability boundaries

The complete experiment pipeline remains separate: model/data preparation, full training launchers, inference, checkpoint management and historical scripts are not included. Training images, captions, model weights and latent caches are not distributed here. The training core receives model components and conditioning from the calling application; the evaluation core receives paired images.

The summary commands reproduce calculations from the included JSON files. The core tests run without downloaded models. Complete SDXL training and generation require the calling application's models, data and workflow. See [CORE.md](CORE.md) for integration requirements and [EVALUATION.md](EVALUATION.md) for the historical evidence limits.

No license has been selected; see [LICENSE_STATUS.md](../LICENSE_STATUS.md). Any change to the public implementation or asset scope should be reflected in this document and both READMEs.
