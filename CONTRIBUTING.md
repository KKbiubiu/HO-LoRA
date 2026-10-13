# Contributing

This repository accepts focused improvements to the core method, image metrics, documentation and recorded summaries. Report an issue with the affected file, the expected correction and supporting evidence.

## Verify a change

Use Python 3.10 or newer from the repository root:

```bash
python3 tools/summarize_results.py --check
python3 -m unittest discover -s tests -p test_results.py -v
```

If the summary utility changes, regenerate the tables with `python3 tools/summarize_results.py` and run both checks again. Keep raw evaluation JSON unchanged unless a source correction is documented; update its provenance when a correction is necessary.

For core changes, install the dependencies described in [CORE.md](docs/CORE.md), then run:

```bash
python -m unittest discover -s tests -p 'test_*_core.py' -v
```

Keep the English and Chinese READMEs consistent when changing availability, setup steps or result interpretation. Describe the change and its verification in the pull request.

## Contribution scope

Keep changes within the two core modules and their documented integration boundary. Full experiment orchestration and image generation remain outside this release. Share only material you have permission to distribute; exclude private data, model weights, credentials and personal machine details. License status is recorded in [LICENSE_STATUS.md](LICENSE_STATUS.md).
