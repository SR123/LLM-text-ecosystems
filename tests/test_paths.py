from pathlib import Path


def test_core_paths_exist():
    root = Path(__file__).resolve().parents[2]
    assert (root / "Nat_Paper").exists()
    assert (root / "GitHub" / "src" / "drift_selection").exists()
    assert (root / "GitHub" / "scripts").exists()


def test_manifest_exists():
    root = Path(__file__).resolve().parents[2]
    assert (root / "GitHub" / "corpora" / "manifests" / "corpora_manifest.yaml").exists()


def test_transformer_pipeline_files_exist():
    root = Path(__file__).resolve().parents[2]
    required = [
        root / "GitHub" / "configs" / "transformer_teacher_student.yaml",
        root / "GitHub" / "configs" / "selected_decoding.yaml",
        root / "GitHub" / "configs" / "tokenizer_conan_doyle.yaml",
        root / "GitHub" / "scripts" / "run_transformer_teacher_train.py",
        root / "GitHub" / "scripts" / "run_transformer_generate_environments.py",
        root / "GitHub" / "scripts" / "run_transformer_student_train.py",
        root / "GitHub" / "scripts" / "run_transformer_evaluation.py",
        root / "GitHub" / "scripts" / "run_transformer_full_pipeline.py",
        root / "GitHub" / "scripts" / "smoke_test_transformer.py",
    ]
    for path in required:
        assert path.exists(), f"missing: {path}"
