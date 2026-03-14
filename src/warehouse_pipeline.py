"""
warehouse_pipeline.py
Builds medallion layers and star-schema style gold tables from processed student data.

Outputs:
  data/silver/student_master.csv
  data/gold/dim_student.csv
  data/gold/dim_subject.csv
  data/gold/dim_semester.csv
  data/gold/fact_student_semester.csv
  data/gold/fact_student_risk.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR = BASE_DIR / "data" / "gold"


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[WH] saved {path} ({df.shape[0]} rows, {df.shape[1]} cols)")


def _series_or_default(df: pd.DataFrame, column: str, default: str = "Unknown") -> pd.Series:
    if column in df.columns:
        return df[column].astype(str)
    return pd.Series([default] * len(df), index=df.index)


def build() -> None:
    students_path = PROCESSED_DIR / "final_student_dataset.csv"
    subjects_path = PROCESSED_DIR / "subject_summary.csv"
    if not students_path.exists() or not subjects_path.exists():
        raise FileNotFoundError("Processed dataset missing. Run ETL first.")

    students = pd.read_csv(students_path)
    subjects = pd.read_csv(subjects_path)
    if students.empty:
        raise ValueError("Processed student dataset is empty.")

    # Silver layer
    silver = students.copy()
    _save(silver, SILVER_DIR / "student_master.csv")

    # Gold dimensions
    dim_student = pd.DataFrame(
        {
            "student_key": np.arange(1, len(silver) + 1),
            "student_id_perf": silver["student_id_perf"].astype(int),
            "gender": _series_or_default(silver, "gender", default="Unknown")
            if "gender" in silver.columns
            else _series_or_default(silver, "gender_x", default="Unknown"),
            "department": _series_or_default(silver, "department", default="Unknown"),
            "career_path": _series_or_default(silver, "career_path", default="Unknown"),
        }
    )
    _save(dim_student, GOLD_DIR / "dim_student.csv")

    dim_subject = subjects.copy().reset_index(drop=True)
    dim_subject.insert(0, "subject_key", np.arange(1, len(dim_subject) + 1))
    _save(dim_subject, GOLD_DIR / "dim_subject.csv")

    semester_cols = sorted(
        [col for col in silver.columns if col.startswith("sgpa_sem_")],
        key=lambda col: int(col.split("_")[-1]),
    )
    semester_numbers = [int(col.split("_")[-1]) for col in semester_cols]
    dim_semester = pd.DataFrame(
        {
            "semester_key": np.arange(1, len(semester_numbers) + 1),
            "semester_no": semester_numbers,
            "semester_label": [f"Semester {n}" for n in semester_numbers],
        }
    )
    _save(dim_semester, GOLD_DIR / "dim_semester.csv")

    # Gold facts
    student_key_lookup = dict(zip(dim_student["student_id_perf"], dim_student["student_key"]))
    sem_key_lookup = dict(zip(dim_semester["semester_no"], dim_semester["semester_key"]))

    semester_facts: list[pd.DataFrame] = []
    for col in semester_cols:
        sem_no = int(col.split("_")[-1])
        part = silver[["student_id_perf", col]].copy()
        part.rename(columns={col: "sgpa"}, inplace=True)
        part["semester_no"] = sem_no
        semester_facts.append(part)

    if semester_facts:
        fact_student_semester = pd.concat(semester_facts, axis=0, ignore_index=True)
        fact_student_semester["student_key"] = fact_student_semester["student_id_perf"].map(student_key_lookup)
        fact_student_semester["semester_key"] = fact_student_semester["semester_no"].map(sem_key_lookup)
        fact_student_semester = fact_student_semester[
            ["student_key", "semester_key", "student_id_perf", "semester_no", "sgpa"]
        ].dropna(subset=["sgpa"])
    else:
        fact_student_semester = pd.DataFrame(
            columns=["student_key", "semester_key", "student_id_perf", "semester_no", "sgpa"]
        )
    _save(fact_student_semester, GOLD_DIR / "fact_student_semester.csv")

    if "attendance_rate" not in silver.columns:
        att_cols = [c for c in ["th_attendance_rate", "pr_attendance_rate"] if c in silver.columns]
        if att_cols:
            silver["attendance_rate"] = silver[att_cols].mean(axis=1)
        else:
            silver["attendance_rate"] = 0.75

    if "placement_readiness_score" not in silver.columns:
        silver["placement_readiness_score"] = (pd.to_numeric(silver["mean_cgpa"], errors="coerce").fillna(0) * 10).clip(0, 100)

    if "stress_numeric" not in silver.columns:
        if "stress_level" in silver.columns:
            stress_map = {"fabulous": 0, "good": 1, "bad": 2, "awful": 3}
            silver["stress_numeric"] = (
                silver["stress_level"].astype(str).str.strip().str.lower().map(stress_map).fillna(1)
            )
        else:
            silver["stress_numeric"] = 1.0

    if "academic_risk_score" not in silver.columns:
        cgpa_part = np.clip((7 - pd.to_numeric(silver["mean_cgpa"], errors="coerce").fillna(0)) / 7.0, 0, 1) * 40
        backlog_part = (
            np.clip(pd.to_numeric(silver["total_backlogs"], errors="coerce").fillna(0) / 10.0, 0, 1) * 30
        )
        stress_part = np.clip(pd.to_numeric(silver["stress_numeric"], errors="coerce").fillna(1) / 3.0, 0, 1) * 15
        att_part = np.clip((0.75 - silver["attendance_rate"]) / 0.75, 0, 1) * 15
        silver["academic_risk_score"] = (cgpa_part + backlog_part + stress_part + att_part).round(2)

    if "at_risk_student" not in silver.columns:
        silver["at_risk_student"] = (
            (pd.to_numeric(silver["mean_cgpa"], errors="coerce").fillna(0) < 5)
            | (pd.to_numeric(silver["total_backlogs"], errors="coerce").fillna(0) > 2)
            | (silver["attendance_rate"] < 0.6)
        )

    fact_student_risk = silver[
        [
            "student_id_perf",
            "mean_cgpa",
            "total_backlogs",
            "stress_numeric",
            "attendance_rate",
            "academic_risk_score",
            "at_risk_student",
            "placement_readiness_score",
        ]
    ].copy()
    fact_student_risk["student_key"] = fact_student_risk["student_id_perf"].map(student_key_lookup)
    fact_student_risk = fact_student_risk[
        [
            "student_key",
            "student_id_perf",
            "mean_cgpa",
            "total_backlogs",
            "stress_numeric",
            "attendance_rate",
            "academic_risk_score",
            "at_risk_student",
            "placement_readiness_score",
        ]
    ]
    _save(fact_student_risk, GOLD_DIR / "fact_student_risk.csv")

    print("[WH] warehouse pipeline completed.")


if __name__ == "__main__":
    build()
