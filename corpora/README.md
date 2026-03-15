# Corpora

This folder stores raw, cleaned, processed, and split datasets for literary/scientific corpora.

## Included corpus slots

- Arthur Conan Doyle (priority)
- Jane Austen
- Charles Darwin
- Mark Twain
- Mary Shelley

## Structure

- `raw/<corpus>/`
- `cleaned/<corpus>/`
- `processed/<corpus>/`
- `splits/<corpus>/`
- `manifests/corpora_manifest.yaml`

## Workflow

1. `scripts/download_corpora.py`
2. `scripts/clean_corpora.py`
3. `scripts/build_splits.py`

Conan Doyle raw files have been imported from prior local workspace assets.
