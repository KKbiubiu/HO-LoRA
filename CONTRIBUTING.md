# Contributing

This repository accepts improvements to documentation, result metadata, summary arithmetic and evaluation explanations. Report an issue with the affected file, the expected correction and supporting evidence.

## Verify a change

Use Python 3.10 or newer from the repository root:

```bash
python3 tools/summarize_results.py --check
python3 -m unittest discover -s tests -v
```

If the summary utility changes, regenerate the tables with `python3 tools/summarize_results.py` and run both checks again. Keep raw evaluation JSON unchanged unless a source correction is documented; update its provenance when a correction is necessary.

Keep the English and Chinese READMEs consistent when changing availability, setup steps or result interpretation. Describe the change and its verification in the pull request.

## Contribution scope

Training, generation and image evaluation implementations are outside this repository's public scope. Share only material you have permission to distribute; exclude private data, model weights, credentials and personal machine details. License status is recorded in [LICENSE_STATUS.md](LICENSE_STATUS.md).
