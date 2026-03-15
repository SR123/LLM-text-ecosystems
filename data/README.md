# Data

Central storage for experiment inputs/outputs and metadata.

## Key paths

- Registry DB: `databases/experiment_index.sqlite`
- Output CSV summaries: `outputs/csv/`
- Run metadata JSON: `outputs/json/`
- Text samples: `outputs/text_samples/`
- Run manifests: `manifests/`

The registry is designed to be checkpoint/resume friendly and mirrored to human-readable CSV exports.
