# Drift and selection in LLM text ecosystems

Code, data, and notebooks accompanying the paper:

> **Drift and selection in LLM text ecosystems**
> Søren Riis, Queen Mary University of London
> arXiv preprint (2026)

## Overview

This repository provides the exact code used to generate all figures and tables in the paper and its appendix. The framework develops an exactly solvable mathematical model of recursive text generation based on variable-order *n*-gram agents, separating two forces acting on the public corpus: **drift** (neutral resampling erodes rare forms) and **selection** (publication filtering reshapes what survives).

The three main theorems characterise:
1. **Theorem 1** (drift and fixed-point polytope): finite-sample Wright–Fisher drift and the complete characterisation of fixed points as circulations on de Bruijn graphs.
2. **Theorem 2** (fixed points under selection): descriptive publication drives the corpus to *n*-shallowness; normative publication sustains deep structure with a KL gap bounded by *L* log₂ *s* bits (optimal).
3. **Theorem 3** (cross-entropy inheritance): later learners trained on the filtered environment recover the public conditional.

## Repository structure

```
LLM-text-ecosystems/
  notebooks/
    active/                 # Paper and appendix experiment notebooks
    basic_theory/           # Core theory notebooks (fixed-point enumeration, diagnostics)
    transformer_experiments/ # Transformer pilot experiments (not in paper; see README)
  scripts/                  # Standalone scripts for figure/table generation and pipelines
  src/drift_selection/      # Python package with core modules
  configs/                  # YAML configuration files
  tests/                    # Unit and smoke tests
  corpora/
    splits/                 # Train/validation/test splits (Doyle, Austen, Darwin)
    manifests/              # Corpus metadata
  figures/
    paper_main/             # Main paper figures
    appendix/               # Appendix figures
  data/
    reference_assets/       # Consolidated tables and plot exports
  appendix/
    tables/                 # LaTeX table sources for appendix
    figures/                # Appendix figure files
  docs/                     # Technical notes and experiment inventory
  web_demos/                # Interactive HTML demonstrations
```

## Notebooks

### Paper and appendix experiments (`notebooks/active/`)

| Notebook | Theorem | Content |
|----------|---------|---------|
| `02_conan_doyle_ngram_reproduction` | — | Conan Doyle trigram recursive loop (Figure 3) |
| `08_selected_decoding_demo` | 2 | Inspection of selected-decoding behaviour |
| `11_theorem1_negative_literary_drift` | 1 | Rare-form thinning under recursive resampling |
| `12_theorem1_positive_standardization` | 1 | Error-correction and normalisation under drift |
| `13_theorem2_positive_artifact_learning` | 2 | Filtered artifacts are easier to imitate (tiling task) |
| `14_theorem2_negative_hidden_steps` | 2 | Omitted intermediate traces damage process-learning |
| `15_theorem3_inheritance_matrix` | 3 | Cross-entropy inheritance across environments |
| `16_arithmetic_full_trace_vs_final_answer` | 2 | Full traces vs final-answer-only corpora (arithmetic) |
| `17_theorem1_vocabulary_drift_trigrams` | 1 | Large-scale vocabulary reduction (Doyle, Austen, Darwin) |
| `18_theorem1_multi_corpus_alpha_sweep` | 1 | Multi-corpus alpha-sweep |
| `19_theorem1_austen_noise_and_standardization` | 1 | Two-sided drift: harmful loss + beneficial standardisation |
| `20_theorem2_process_learning_workflow` | 2 | Process-trace vs artifact-only training (Task A & B) |
| `21_theorem2_artifact_learning_workflow` | 2 | Artifact-only environment on tiling tasks |
| `22_theorem2_taskB_environment_modes_workflow` | 2 | Three arithmetic environments × two deployment modes |
| `23_theorem2_selected_publication` | 2 | Matched exact Theorem 2 diagnostics (Figure 2/8) |
| `24_theorem1_synthetic_regeneration` | 1 | Synthetic support drift in 3-gram systems |
| `25_general_ngram_regeneration_lab` | 1–2 | General lab for *n*-gram regeneration with configurable parameters |
| `26_tutorial6_artifact_vs_process_learning` | 2 | Teaching notebook: artifact vs process learning |

### Core theory (`notebooks/basic_theory/`)

| Notebook | Content |
|----------|---------|
| `ngram_population_large_scale_theorem1` | Large-scale neutral *n*-gram population maps with convergence diagnostics |
| `ngram_theorem2_strong_diagnostics_teaching` | Interactive Theorem 2 teaching notebook |
| `count_extreme_points` | Extreme-point enumeration for Table 2 (de Bruijn cycle counting) |

### Transformer experiments (`notebooks/transformer_experiments/`)

Pilot transformer-based experiments exploring whether the *n*-gram theory's predictions (artifact learning vs process learning) also hold for small neural language models trained on the Conan Doyle corpus. These were not included in the paper. See the [README](notebooks/transformer_experiments/README.md) in that folder.

## Key scripts

| Script | Purpose |
|--------|---------|
| `build_theorem2_information_tutorial_case.py` | Generates the matched exact experiment (Section 5.3) |
| `build_conan_doyle_figure3_reproduction.py` | Reproduces Figure 3 |
| `build_all_figures.py` | Master script to regenerate all paper figures |
| `build_appendix_tables.py` | Generates appendix LaTeX tables |
| `download_corpora.py` | Downloads Project Gutenberg source texts |
| `clean_corpora.py` | Normalises raw text corpora |
| `build_splits.py` | Creates train/validation/test splits |

## Installation

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate drift_and_selection
```

### Requirements

- Python ≥ 3.11
- numpy, pandas, matplotlib, tqdm, PyYAML
- torch ≥ 2.2 (for transformer experiments only)
- pytest (for tests)

## Corpora

The text corpora are public-domain works from Project Gutenberg:
- **Arthur Conan Doyle** (~4M tokens, ~55k vocabulary) — primary running example
- **Jane Austen** (~601k tokens, ~14k vocabulary) — supplementary
- **Charles Darwin** (~356k tokens, ~15.6k vocabulary) — supplementary

Pre-built train/validation/test splits are included in `corpora/splits/`. To download and rebuild from raw Gutenberg sources: `python scripts/download_corpora.py && python scripts/clean_corpora.py && python scripts/build_splits.py`.

## Interactive demos

The `web_demos/` folder contains four self-contained HTML demonstrations:
- **Drift (unigram):** visualises Wright–Fisher drift on token frequencies
- **Absorbing Markov:** demonstrates absorbing-state dynamics
- **N-gram lookahead:** shows how lookahead changes the effective publication law
- **Selected decoding:** illustrates selection-filtered text generation

Open `web_demos/index.html` in a browser.

## Citation

```bibtex
@article{riis2026drift,
  title={Drift and selection in {LLM} text ecosystems},
  author={Riis, S{\o}ren},
  year={2026},
  journal={arXiv preprint}
}
```

## License

MIT License. See [LICENSE](LICENSE).
