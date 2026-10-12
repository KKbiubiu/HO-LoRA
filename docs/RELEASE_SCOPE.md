# Release scope

HO-LoRA's public repository presents the project and its recorded results. Its runnable component is the standard-library utility for generating and verifying summary tables.

## Included

- English and Chinese project overviews.
- Selected generated-image comparisons and source hashes.
- Six recorded evaluation JSON files and generated summary tables.
- Evaluation prompts, recorded v12 settings and provenance.
- Method and evaluation documentation.
- Summary utility, consistency tests and continuous-integration checks.

## Availability boundaries

The training, image generation and image evaluation implementations are not public. Training images, captions, model weights, latent caches and checkpoints are not distributed here. The public repository provides no model download or generation API.

The summary commands in the README reproduce calculations from the included JSON files. They cannot reproduce training or generate the comparison images from this repository alone. Historical reproducibility limits are described in [EVALUATION.md](EVALUATION.md).

No license has been selected; see [LICENSE_STATUS.md](../LICENSE_STATUS.md). Any change to the public implementation or asset scope should be reflected in this document and both READMEs.
