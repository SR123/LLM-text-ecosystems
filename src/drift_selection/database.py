from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


def _table_columns(con: sqlite3.Connection, table_name: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(r[1]) for r in rows}


def _ensure_columns(con: sqlite3.Connection, table_name: str, required: dict[str, str]) -> None:
    existing = _table_columns(con, table_name)
    for col, col_type in required.items():
        if col not in existing:
            con.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")


def init_experiment_registry(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT,
                timestamp_start TEXT,
                timestamp_end TEXT,
                git_version TEXT,
                script_name TEXT,
                corpus TEXT,
                model_family TEXT,
                selection_rule TEXT,
                seed INTEGER,
                config_path TEXT,
                output_dir TEXT,
                metrics_summary TEXT,
                status TEXT,
                metadata_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                artifact_type TEXT,
                artifact_path TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                metric_name TEXT,
                metric_value REAL,
                split TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS corpora (
                corpus_id INTEGER PRIMARY KEY AUTOINCREMENT,
                corpus_name TEXT UNIQUE,
                manifest_path TEXT,
                note TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS configs (
                config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_name TEXT UNIQUE,
                config_path TEXT,
                config_hash TEXT
            )
            """
        )
        # Lightweight schema migration for older local DBs created before
        # metadata_json/split columns were introduced.
        _ensure_columns(
            con,
            "runs",
            {
                "metadata_json": "TEXT",
            },
        )
        _ensure_columns(
            con,
            "metrics",
            {
                "split": "TEXT",
            },
        )
        con.commit()


def register_run(
    db_path: Path,
    run_name: str,
    script_or_notebook: str,
    config_path: str,
    metadata: dict,
) -> int:
    init_experiment_registry(db_path)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO runs (
                run_name,timestamp_start,timestamp_end,git_version,script_name,corpus,model_family,
                selection_rule,seed,config_path,output_dir,metrics_summary,status,metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                metadata.get("timestamp_start"),
                metadata.get("timestamp_end"),
                metadata.get("git_version"),
                script_or_notebook,
                metadata.get("corpus"),
                metadata.get("model_family"),
                metadata.get("selection_rule"),
                metadata.get("seed"),
                config_path,
                metadata.get("output_dir"),
                metadata.get("metrics_summary"),
                metadata.get("status", "running"),
                json.dumps(metadata, sort_keys=True),
            ),
        )
        con.commit()
        return int(cur.lastrowid)


def log_run_metric(
    db_path: Path,
    run_id: int,
    metric_name: str,
    metric_value: float,
    split: str | None = None,
) -> None:
    init_experiment_registry(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO metrics (run_id,metric_name,metric_value,split) VALUES (?, ?, ?, ?)",
            (run_id, metric_name, float(metric_value), split),
        )
        con.commit()


def log_artifact(
    db_path: Path,
    run_id: int,
    artifact_type: str,
    path: str,
) -> None:
    init_experiment_registry(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO artifacts (run_id,artifact_type,artifact_path) VALUES (?, ?, ?)",
            (run_id, artifact_type, path),
        )
        con.commit()


def export_registry_csv(
    db_path: Path,
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    init_experiment_registry(db_path)
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT run_id,run_name,timestamp_start,timestamp_end,corpus,model_family,selection_rule,seed,status,output_dir,metrics_summary FROM runs ORDER BY run_id"
        ).fetchall()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id",
            "run_name",
            "timestamp_start",
            "timestamp_end",
            "corpus",
            "model_family",
            "selection_rule",
            "seed",
            "status",
            "output_dir",
            "metrics_summary",
        ])
        for row in rows:
            w.writerow(row)


class ExperimentDB:
    """Backward-compatible OOP wrapper around function-based registry helpers."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init_schema(self) -> None:
        init_experiment_registry(self.db_path)

    def start_run(self, payload: dict[str, Any]) -> int:
        return register_run(
            db_path=self.db_path,
            run_name=payload.get("run_name", "unnamed"),
            script_or_notebook=payload.get("script_name", "unknown"),
            config_path=payload.get("config_path", ""),
            metadata=payload,
        )

    def end_run(self, run_id: int, timestamp_end: str, status: str = "completed", metrics_summary: str | None = None) -> None:
        init_experiment_registry(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "UPDATE runs SET timestamp_end=?, status=?, metrics_summary=COALESCE(?, metrics_summary) WHERE run_id=?",
                (timestamp_end, status, metrics_summary, run_id),
            )
            con.commit()

    def add_artifact(self, run_id: int, artifact_type: str, artifact_path: str) -> None:
        log_artifact(self.db_path, run_id, artifact_type, artifact_path)

    def add_metric(self, run_id: int, metric_name: str, metric_value: float) -> None:
        log_run_metric(self.db_path, run_id, metric_name, metric_value)

    def upsert_corpus(self, corpus_name: str, manifest_path: str, note: str = "") -> None:
        init_experiment_registry(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO corpora (corpus_name,manifest_path,note) VALUES (?, ?, ?) "
                "ON CONFLICT(corpus_name) DO UPDATE SET manifest_path=excluded.manifest_path, note=excluded.note",
                (corpus_name, manifest_path, note),
            )
            con.commit()

    def upsert_config(self, config_name: str, config_path: str, config_hash: str) -> None:
        init_experiment_registry(self.db_path)
        with sqlite3.connect(self.db_path) as con:
            con.execute(
                "INSERT INTO configs (config_name,config_path,config_hash) VALUES (?, ?, ?) "
                "ON CONFLICT(config_name) DO UPDATE SET config_path=excluded.config_path, config_hash=excluded.config_hash",
                (config_name, config_path, config_hash),
            )
            con.commit()

    def export_registry_csv(self, csv_path: Path) -> None:
        export_registry_csv(self.db_path, csv_path)
