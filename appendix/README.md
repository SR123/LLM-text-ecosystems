# Appendix Companion

This folder contains the live guided appendix used as the reader-facing companion to the Nature manuscript, plus archived older appendix releases.

## Live guided appendix

- `Appendix_Guided_v0_73.tex`
- `Appendix_Guided_v0_73.pdf`

The old `appendix_latest.tex` and `appendix_latest.pdf` alias files have been retired and archived under `archive/legacy_aliases/`.

## Compile

```bash
cd GitHub/appendix
./compile_appendix.sh
```

The script defaults to the current versioned guided appendix and writes only the versioned output PDF.

## Structure

- `sections/`: theorem/sign experiment sections loaded by `\\input{...}`
- `figures/`: all appendix image files used by the TeX sources
- `figures/generated/`: generated theorem-specific figure panels
- `tables/`: appendix table sources
- `archive/`: older appendix releases and their build artifacts

## Current generated assets

Theorem 1:

- `figures/generated/figure_appx_theorem1_vocab_retention_multi_corpus.pdf`
- `figures/generated/figure_appx_theorem1_trigram_retention_multi_corpus.pdf`
- `figures/generated/figure_appx_theorem1_active_vocab_threshold_multi_corpus.pdf`
- `figures/generated/figure_appx_theorem1_austen_orthographic_generation_panels.pdf`
- `figures/generated/figure_appx_theorem1_austen_orthographic_rate_sweep.pdf`

Theorem 2:

- `figures/generated/figure_appx_theorem2_taskA_process_vs_artifact_metrics.pdf`
- `figures/generated/figure_appx_theorem2_taskB_direct_metrics.pdf`
- `figures/generated/figure_appx_theorem2_taskB_process_metrics.pdf`
- `figures/generated/figure_appx_theorem2_taskB_direct_contamination.pdf`

Theorem 3:

- `figures/generated/figure_appx_theorem3_cross_entropy_inheritance.pdf`
- `figures/generated/figure_appx_theorem3_iterated_environment_extension.pdf`

Tables:

- `tables/table_appx_theorem1_multi_corpus_endpoints.tex`
- `tables/table_appx_theorem1_austen_orthographic_endpoints.tex`
- `tables/table_appx_theorem2_negative_protocol.tex`
- `tables/table_appx_theorem2_positive_protocol.tex`
- `tables/table_appx_theorem3_cross_entropy_results.tex`
- `tables/table_appx_theorem3_iterated_environment_extension.tex`

## Note

The appendix figure files are duplicated under `../data/reference_assets/appendix_figures/` for upload convenience, but the TeX sources compile directly from `appendix/figures/`.

Older appendix releases, together with the retired `appendix_latest` alias files, now live under `archive/`.
