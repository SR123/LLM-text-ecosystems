from pathlib import Path
import sys
import tempfile

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from drift_selection.database import ExperimentDB


def test_database_init_and_run_insert():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "exp.sqlite"
        db = ExperimentDB(db_path)
        db.init_schema()
        run_id = db.start_run({
            "run_name": "unit",
            "timestamp_start": "2026-01-01T00:00:00Z",
            "script_name": "test",
            "corpus": "toy",
            "model_family": "ngram",
            "selection_rule": "neutral",
            "seed": 1,
            "config_path": "cfg.yaml",
            "output_dir": "out",
            "status": "running",
        })
        assert run_id >= 1
        db.end_run(run_id, "2026-01-01T00:10:00Z", status="completed")
        csv_path = Path(tmp) / "registry.csv"
        db.export_registry_csv(csv_path)
        assert csv_path.exists()
