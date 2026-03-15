# Transformer experiments: artifact learning versus process learning

These notebooks contain the transformer-based experiments that complement the
*n*-gram theory in the main paper. They were not included directly in the paper
or appendix, but they illustrate the same drift, selection, and inheritance
phenomena using small neural language models trained on the Conan Doyle corpus.

## Experiment overview

The central question is whether the artifact-learning versus process-learning
distinction identified by the *n*-gram theory (Theorem 2) also manifests when
the generator and learner are small transformers rather than *n*-gram models.

**Setup.** A tiny transformer "teacher" is trained on the Conan Doyle fiction
corpus. It then generates synthetic text under two regimes:

1. **Descriptive (neutral):** the teacher publishes its own samples unfiltered.
2. **Normative (selected):** a lookahead verifier filters the teacher's output,
   keeping only traces that survive a prescribed number of continuation steps.

A "student" transformer is then trained on each filtered environment and
evaluated on held-out prompts. The diagnostics mirror those of the *n*-gram
matched experiments in Section 5 of the appendix.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `07_transformer_conan_doyle_teacher_student.ipynb` | End-to-end pipeline: train teacher, generate environments, train student, evaluate |
| `09_transformer_metrics_and_figures.ipynb` | Rebuild analysis figures from saved artifacts (no retraining needed) |
| `10_transformer_ablation_pilot.ipynb` | Ablation over selected-decoding parameters (horizon, top-k, beam width) |

## Relation to the paper

These experiments provide evidence that the *n*-gram results are not artifacts
of the *n*-gram model class. The same qualitative pattern — descriptive
publication compresses structure while normative publication preserves it —
appears with transformer generators. A full transformer-scale study is left
for future work; these notebooks document the pilot exploration.
