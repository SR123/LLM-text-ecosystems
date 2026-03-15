# Transformer Conan Doyle Experiment

## Objective

Train a tiny GPT-style teacher on Conan Doyle, generate neutral/selected synthetic environments,
train matched students on original vs synthetic environments, and evaluate policy inheritance.

## Pipeline scripts

1. `python GitHub/scripts/run_transformer_teacher_train.py --root . --config GitHub/configs/transformer_teacher_student.yaml`
2. `python GitHub/scripts/run_transformer_generate_environments.py --root . --config GitHub/configs/transformer_teacher_student.yaml --mode pilot`
3. `python GitHub/scripts/run_transformer_student_train.py --root . --config GitHub/configs/transformer_teacher_student.yaml --mode pilot`
4. `python GitHub/scripts/run_transformer_evaluation.py --root . --config GitHub/configs/transformer_teacher_student.yaml --mode pilot`

Or run end-to-end:

`python GitHub/scripts/run_transformer_full_pipeline.py --root . --config GitHub/configs/transformer_teacher_student.yaml --mode pilot`

## Required dependency note

`sentencepiece` must be installed to train/load the shared tokenizer.

## Main outputs

- `GitHub/data/outputs/transformer_conan_doyle/teacher/`
- `GitHub/data/outputs/transformer_conan_doyle/environments/`
- `GitHub/data/outputs/transformer_conan_doyle/students/`
- `GitHub/data/outputs/transformer_conan_doyle/evaluation/`
- `GitHub/figures/paper_main/transformer_teacher_student_bars.pdf`
