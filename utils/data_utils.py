from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
BRONZE_DIR = BASE_DIR / "data" / "bronze"
SILVER_DIR = BASE_DIR / "data" / "silver"
GOLD_DIR = BASE_DIR / "data" / "gold"
ETL_STATUS_FILE = PROCESSED_DIR / "etl_status.json"
DATASET_HISTORY_FILE = PROCESSED_DIR / "dataset_history.jsonl"
DEFAULT_DB_URL = "mysql://root:9963@localhost:3306/studentanalyticdashboard"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL).strip()
DB_TABLE_STUDENTS = os.getenv("DB_TABLE_STUDENTS", "final_student_dataset").strip()
DB_TABLE_SUBJECTS = os.getenv("DB_TABLE_SUBJECTS", "subject_summary").strip()
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

STUDY_MAP = {
    "0 - 30 minute": 0.25,
    "30 - 60 minute": 0.75,
    "1 - 2 hour": 1.50,
    "2 - 3 hour": 2.50,
    "3 - 4 hour": 3.50,
    "more than 4 hour": 5.00,
}

STRESS_MAP = {
    "fabulous": 0,
    "good": 1,
    "bad": 2,
    "awful": 3,
}

TRAVEL_MAP = {
    "0 - 30 minutes": 15,
    "30 - 60 minutes": 45,
    "1 - 1.30 hour": 75,
    "1.30 - 2 hour": 105,
    "2 - 2.30 hour": 135,
    "2.30 - 3 hour": 165,
    "more than 3 hour": 195,
}

SOCIAL_MAP = {
    "0 minute": 0.00,
    "1 - 30 minute": 0.25,
    "30 - 60 minute": 0.75,
    "30 - 60 minutes": 0.75,
    "1 - 1.30 hour": 1.25,
    "1.30 - 2 hour": 1.75,
    "more than 2 hour": 2.50,
}


def _safe_table_name(name: str, fallback: str) -> str:
    cleaned = (name or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_]+", cleaned):
        return cleaned
    return fallback


def _normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _normalized_database_url() -> str:
    url = DATABASE_URL.strip().strip('"').strip("'")
    if url.startswith("mysql://"):
        return url.replace("mysql://", "mysql+pymysql://", 1)
    return url


@st.cache_resource(show_spinner=False)
def get_db_engine():
    url = _normalized_database_url()
    if not url:
        return None
    try:
        from sqlalchemy import create_engine

        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return engine
    except Exception:
        return None


def get_database_health() -> tuple[bool, str]:
    if not DATABASE_URL:
        return False, "DATABASE_URL is not configured."
    engine = get_db_engine()
    if engine is None:
        return False, "MySQL connection unavailable. Check URL, service, and dependencies."
    student_table = _safe_table_name(DB_TABLE_STUDENTS, "final_student_dataset")
    subject_table = _safe_table_name(DB_TABLE_SUBJECTS, "subject_summary")
    return True, f"Connected. Using tables: {student_table}, {subject_table}."


def sync_processed_to_db() -> tuple[bool, str]:
    engine = get_db_engine()
    if engine is None:
        return False, "Database connection is not available."

    student_table = _safe_table_name(DB_TABLE_STUDENTS, "final_student_dataset")
    subject_table = _safe_table_name(DB_TABLE_SUBJECTS, "subject_summary")

    student_path = PROCESSED_DIR / "final_student_dataset.csv"
    subject_path = PROCESSED_DIR / "subject_summary.csv"
    if not student_path.exists() or not subject_path.exists():
        return False, "Processed CSV files are missing. Run ETL first."

    student_df = pd.read_csv(student_path)
    subject_df = pd.read_csv(subject_path)

    try:
        student_df.to_sql(student_table, engine, if_exists="replace", index=False, chunksize=1000)
        subject_df.to_sql(subject_table, engine, if_exists="replace", index=False, chunksize=1000)
    except Exception as exc:
        return False, f"DB sync failed: {exc}"

    return (
        True,
        (
            f"Synced to MySQL ({student_table}: {len(student_df)} rows, "
            f"{subject_table}: {len(subject_df)} rows)."
        ),
    )


def _as_series(values: object, length: int) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    if values is None:
        return pd.Series([np.nan] * length)
    return pd.Series(values)


def _to_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    out = pd.to_numeric(df[column], errors="coerce")
    return out.fillna(default)


def _map_text_column(
    df: pd.DataFrame, source_col: str, mapping: dict[str, float], fallback: float
) -> pd.Series:
    if source_col not in df.columns:
        return pd.Series([fallback] * len(df), index=df.index)
    cleaned = df[source_col].astype(str).str.strip().str.lower()
    return cleaned.map(mapping).fillna(fallback)


def _yes_no_binary(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.isin({"yes", "y", "true", "1"}).astype(int)


def _extract_percent(series: pd.Series, default: float = 50.0) -> pd.Series:
    cleaned = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce").fillna(default)


def _risk_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row.get("attendance_rate", 0.0) < 0.70:
        reasons.append("Low attendance")
    if row.get("stress_numeric", 0.0) >= 2.0:
        reasons.append("High stress")
    if row.get("total_backlogs", 0.0) > 2.0:
        reasons.append("Multiple backlogs")
    if row.get("mean_cgpa", 0.0) < 5.5:
        reasons.append("Low CGPA")
    if row.get("study_hours_day", 0.0) < 1.0:
        reasons.append("Low study time")
    if not reasons:
        return "Composite academic risk"
    return ", ".join(reasons[:2])


def infer_skill_gaps(row: pd.Series) -> list[str]:
    gaps: list[str] = []
    if row.get("technical_skill_score", 0.0) < 65:
        gaps.extend(["Machine Learning", "SQL"])
    if row.get("theory_gp_mean", 0.0) < 6:
        gaps.append("Programming Fundamentals")
    if row.get("practical_marks_mean", 0.0) < 60:
        gaps.append("Hands-on Project Building")
    if row.get("study_hours_day", 0.0) < 2:
        gaps.append("Study Consistency")
    if row.get("social_hours", 0.0) > 2:
        gaps.append("Focus and Time Management")

    deduped: list[str] = []
    for gap in gaps:
        if gap not in deduped:
            deduped.append(gap)
    return deduped[:4] if deduped else ["System Design", "Communication", "Advanced SQL"]


def _recommend_career(row: pd.Series) -> str:
    tech = row.get("technical_skill_score", 0.0)
    startup = row.get("startup_potential_score", 0.0)
    cgpa = row.get("mean_cgpa", 0.0)
    readiness = row.get("placement_readiness_score", 0.0)

    if startup >= 72 and row.get("technical_projects", 0.0) >= 2:
        return "Entrepreneur"
    if tech >= 72 and cgpa >= 7.0:
        return "Software Engineer"
    if tech >= 62 and readiness >= 65:
        return "Data Analyst"
    if row.get("average_marks", 0.0) >= 72:
        return "Business Analyst"
    return "Operations Specialist"


def get_semester_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith("sgpa_sem_")]
    return sorted(cols, key=lambda col: int(col.split("_")[-1]))


def prepare_student_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "student_id_perf",
        "latest_cgpa",
        "mean_cgpa",
        "mean_sgpa",
        "total_backlogs",
        "th_attendance_rate",
        "pr_attendance_rate",
        "theory_gp_mean",
        "practical_marks_mean",
        "study_hours_day",
        "stress_numeric",
        "travel_mins",
        "social_hours",
        "average_marks",
        "academic_risk_score",
        "technical_projects",
        "tech_quiz",
        "olympiads_qualified",
        "engg_coaching",
        "miscellany_tech_events",
    ]
    for col in numeric_cols:
        out[col] = _to_numeric(out, col)

    if out["student_id_perf"].fillna(0).eq(0).all():
        out["student_id_perf"] = pd.Series(np.arange(1, len(out) + 1), index=out.index)

    mapped_study = _map_text_column(out, "daily_studing_time", STUDY_MAP, fallback=1.50)
    out["study_hours_day"] = out["study_hours_day"].replace(0, np.nan).fillna(mapped_study)

    mapped_stress = _map_text_column(out, "stress_level", STRESS_MAP, fallback=1.0)
    out["stress_numeric"] = out["stress_numeric"].replace(0, np.nan).fillna(mapped_stress)

    mapped_travel = _map_text_column(out, "travelling_time", TRAVEL_MAP, fallback=45.0)
    out["travel_mins"] = out["travel_mins"].replace(0, np.nan).fillna(mapped_travel)

    mapped_social = _map_text_column(out, "social_medai_video", SOCIAL_MAP, fallback=1.0)
    out["social_hours"] = out["social_hours"].replace(0, np.nan).fillna(mapped_social)

    for attendance_col in ["th_attendance_rate", "pr_attendance_rate"]:
        if attendance_col in out.columns and out[attendance_col].max() > 1:
            out[attendance_col] = out[attendance_col] / 100.0

    available_attendance = [
        col for col in ["th_attendance_rate", "pr_attendance_rate"] if col in out.columns
    ]
    if available_attendance:
        out["attendance_rate"] = out[available_attendance].mean(axis=1).fillna(0.75)
    elif "attendance_rate" in out.columns:
        out["attendance_rate"] = _to_numeric(out, "attendance_rate", default=0.75)
        if out["attendance_rate"].max() > 1:
            out["attendance_rate"] = out["attendance_rate"] / 100.0
    else:
        out["attendance_rate"] = 0.75

    if "average_marks" not in out.columns or out["average_marks"].eq(0).all():
        mark_cols = [col for col in ["10th_mark", "12th_mark", "college_mark"] if col in out.columns]
        if mark_cols:
            out["average_marks"] = out[mark_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        else:
            out["average_marks"] = out["mean_cgpa"] * 10

    internship_source = _as_series(out.get("certification_course"), len(out))
    out["internship_participation"] = _yes_no_binary(internship_source)

    tech_raw = (
        np.clip(out["theory_gp_mean"], 0, 10) * 6.0
        + np.clip(out["practical_marks_mean"], 0, 100) * 0.25
        + np.clip(out["technical_projects"], 0, 10) * 3.0
        + np.clip(out["tech_quiz"], 0, 10) * 2.5
        + np.clip(out["olympiads_qualified"], 0, 10) * 1.5
    )
    out["technical_skill_score"] = np.clip((tech_raw / 155.0) * 100.0, 0, 100).round(2)

    willingness = _extract_percent(
        _as_series(out.get("willingness_to_pursue_a_career_based_on_their_degree"), len(out))
    )
    part_time_source = _as_series(out.get("part_time_job"), len(out))
    part_time = _yes_no_binary(part_time_source)
    startup_raw = (
        out["technical_projects"] * 8.0
        + out["tech_quiz"] * 5.0
        + out["miscellany_tech_events"] * 5.0
        + part_time * 20.0
        + willingness * 0.30
        + np.clip(out["study_hours_day"], 0, 5) * 7.0
    )
    out["startup_potential_score"] = np.clip(startup_raw, 0, 100).round(2)

    if "academic_risk_score" not in out.columns or out["academic_risk_score"].eq(0).all():
        out["academic_risk_score"] = (
            np.clip((7 - out["mean_cgpa"]) / 7.0, 0, 1) * 40.0
            + np.clip(out["total_backlogs"] / 10.0, 0, 1) * 30.0
            + np.clip(out["stress_numeric"] / 3.0, 0, 1) * 15.0
            + np.clip((0.75 - out["attendance_rate"]) / 0.75, 0, 1) * 15.0
        )
    out["academic_risk_score"] = np.clip(out["academic_risk_score"], 0, 100).round(2)

    if "at_risk_student" not in out.columns:
        out["at_risk_student"] = (
            (out["mean_cgpa"] < 5.0)
            | (out["total_backlogs"] > 2)
            | (out["attendance_rate"] < 0.60)
            | ((out["stress_numeric"] >= 2) & (out["study_hours_day"] < 1))
        )
    out["at_risk_student"] = out["at_risk_student"].astype(bool)

    out["risk_level"] = pd.cut(
        out["academic_risk_score"],
        bins=[-0.1, 40, 70, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    out["risk_reason"] = out.apply(_risk_reason, axis=1)

    out["placement_readiness_score"] = (
        out["mean_cgpa"] * 10.0 * 0.35
        + out["attendance_rate"] * 100.0 * 0.25
        + out["technical_skill_score"] * 0.25
        + out["internship_participation"] * 100.0 * 0.10
        + np.clip(out["study_hours_day"] / 4.0, 0, 1) * 100.0 * 0.05
    )
    out["placement_readiness_score"] = np.clip(
        out["placement_readiness_score"], 0, 100
    ).round(2)

    out["sleep_hours"] = (
        8.2
        - (out["stress_numeric"] * 0.5)
        - (out["social_hours"] * 0.2)
        - (out["travel_mins"] / 180.0)
    )
    out["sleep_hours"] = np.clip(out["sleep_hours"], 4.0, 9.0).round(2)

    out["study_efficiency"] = (
        out["mean_cgpa"] / np.where(out["study_hours_day"] <= 0, np.nan, out["study_hours_day"])
    )
    out["study_efficiency"] = out["study_efficiency"].replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

    out["career_path"] = out.apply(_recommend_career, axis=1)
    out["skill_gaps"] = out.apply(lambda row: infer_skill_gaps(row), axis=1)

    student_ids = _to_numeric(out, "student_id_perf", default=0).astype(int).astype(str)
    out["student_label"] = "Student " + student_ids + " (ID " + student_ids + ")"
    return out


@st.cache_data(show_spinner=False)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = get_db_engine()
    student_table = _safe_table_name(DB_TABLE_STUDENTS, "final_student_dataset")
    subject_table = _safe_table_name(DB_TABLE_SUBJECTS, "subject_summary")

    if engine is not None:
        try:
            df = pd.read_sql_query(f"SELECT * FROM `{student_table}`", engine)
            subject_df = pd.read_sql_query(f"SELECT * FROM `{subject_table}`", engine)
            if not df.empty and not subject_df.empty:
                return prepare_student_dataset(df), subject_df
        except Exception:
            # Fall back to file-based loading if DB tables are unavailable.
            pass

    df = pd.read_csv(PROCESSED_DIR / "final_student_dataset.csv")
    subject_df = pd.read_csv(PROCESSED_DIR / "subject_summary.csv")
    return prepare_student_dataset(df), subject_df


@st.cache_resource(show_spinner=False)
def train_models_cached(_df: pd.DataFrame) -> dict:
    from ml_models import run_all_models

    return run_all_models(_df)


def predict_student_cgpa(student_row: pd.Series, ml_results: dict) -> float:
    regression = ml_results.get("regression", {})
    metrics = regression.get("metrics", {})
    models = regression.get("models", {})
    features = regression.get("features", [])
    if not metrics or not models or not features:
        return float(student_row.get("mean_cgpa", 0.0))

    best_model_name = max(metrics, key=lambda name: metrics[name].get("R2", -1))
    model = models[best_model_name]
    sample = pd.DataFrame([{feature: student_row.get(feature, 0.0) for feature in features}])
    sample = sample.fillna(sample.median(numeric_only=True)).fillna(0)
    prediction = model.predict(sample)[0]
    return float(np.clip(prediction, 0, 10))


def compute_overview_insights(df: pd.DataFrame, subject_df: pd.DataFrame) -> list[str]:
    insights: list[str] = []

    high_study = df.loc[df["study_hours_day"] > 3, "mean_cgpa"].mean()
    low_study = df.loc[df["study_hours_day"] <= 3, "mean_cgpa"].mean()
    if pd.notna(high_study) and pd.notna(low_study) and low_study > 0:
        lift = ((high_study - low_study) / low_study) * 100
        insights.append(
            f"Students studying more than 3 hours/day show about {lift:.1f}% higher CGPA."
        )

    if not subject_df.empty and "fail_rate" in subject_df.columns:
        hardest = subject_df.sort_values("fail_rate", ascending=False).iloc[0]
        insights.append(
            f"{hardest['subject']} has the highest failure rate at {hardest['fail_rate'] * 100:.1f}%."
        )

    internship_readiness = (
        df.groupby("internship_participation")["placement_readiness_score"].mean().to_dict()
    )
    with_intern = internship_readiness.get(1, np.nan)
    without_intern = internship_readiness.get(0, np.nan)
    if pd.notna(with_intern) and pd.notna(without_intern):
        delta = with_intern - without_intern
        insights.append(
            f"Internship participation is linked with a {delta:.1f}-point increase in placement readiness."
        )

    return insights


def get_student_lookup(df: pd.DataFrame) -> dict[str, int]:
    pairs = zip(df["student_label"], df.index, strict=False)
    return {label: int(idx) for label, idx in pairs}


def append_dataset_history(
    action: str,
    source: str,
    status: str = "Success",
    details: dict[str, object] | None = None,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %I:%M:%S %p"),
        "action": action,
        "source": source,
        "status": status,
        "details": details or {},
    }
    with DATASET_HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=True) + "\n")


def get_dataset_history(limit: int = 200) -> pd.DataFrame:
    columns = ["Timestamp", "Action", "Source", "Status", "Details"]
    if not DATASET_HISTORY_FILE.exists():
        return pd.DataFrame(columns=columns)

    events: list[dict[str, object]] = []
    with DATASET_HISTORY_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not events:
        return pd.DataFrame(columns=columns)

    recent = list(reversed(events[-max(limit, 1) :]))
    rows = []
    for event in recent:
        details = event.get("details", {})
        rows.append(
            {
                "Timestamp": str(event.get("timestamp", "")),
                "Action": str(event.get("action", "")),
                "Source": str(event.get("source", "")),
                "Status": str(event.get("status", "")),
                "Details": json.dumps(details, ensure_ascii=True),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_subject_summary_from_uploaded(
    uploaded_df: pd.DataFrame, prepared_df: pd.DataFrame
) -> pd.DataFrame:
    renamed = uploaded_df.copy()
    renamed.columns = [_normalize_col_name(col) for col in renamed.columns]

    subject_col = None
    for col in ["subject", "subject_name", "subjectname"]:
        if col in renamed.columns:
            subject_col = col
            break

    if subject_col is not None:
        score_col = None
        for col in ["theoryagggradepoint", "theory_gp", "subject_score", "marks", "score"]:
            if col in renamed.columns:
                score_col = col
                break
        if score_col is not None:
            scores = pd.to_numeric(renamed[score_col], errors="coerce")
            # Convert marks scale to grade-point scale if needed.
            if scores.dropna().gt(10).mean() > 0.5:
                scores = scores / 10.0

            grouped = (
                pd.DataFrame({"subject": renamed[subject_col].astype(str), "score_gp": scores})
                .dropna(subset=["score_gp"])
                .groupby("subject")
                .agg(avg_theory_gp=("score_gp", "mean"), student_count=("score_gp", "count"))
                .reset_index()
            )
            if not grouped.empty:
                grouped["avg_practical"] = grouped["avg_theory_gp"] * 10
                grouped["fail_rate"] = (grouped["avg_theory_gp"] < 4).astype(float)
                grouped["difficulty_score"] = (
                    (1 - grouped["avg_theory_gp"].clip(0, 10) / 10.0) * 50
                    + grouped["fail_rate"] * 50
                )
                return grouped[
                    ["subject", "avg_theory_gp", "avg_practical", "student_count", "fail_rate", "difficulty_score"]
                ]

    base_avg_gp = float(prepared_df["mean_cgpa"].mean()) if "mean_cgpa" in prepared_df.columns else 6.0
    fail_rate = float((prepared_df["mean_cgpa"] < 5).mean()) if "mean_cgpa" in prepared_df.columns else 0.2
    fallback = pd.DataFrame(
        [
            {
                "subject": "General Performance",
                "avg_theory_gp": base_avg_gp,
                "avg_practical": base_avg_gp * 10,
                "student_count": len(prepared_df),
                "fail_rate": fail_rate,
                "difficulty_score": ((1 - min(base_avg_gp, 10) / 10.0) * 50) + fail_rate * 50,
            }
        ]
    )
    return fallback


def activate_uploaded_dataset(upload_path: Path) -> tuple[bool, str]:
    if not upload_path.exists():
        return False, f"Uploaded file not found: {upload_path}"

    suffix = upload_path.suffix.lower()
    try:
        if suffix == ".csv":
            uploaded_df = pd.read_csv(upload_path)
        elif suffix in {".xlsx", ".xls"}:
            uploaded_df = pd.read_excel(upload_path)
        else:
            return False, "Unsupported file type. Use CSV or XLSX."
    except Exception as exc:
        return False, f"Failed to read uploaded dataset: {exc}"

    if uploaded_df.empty:
        return False, "Uploaded dataset is empty."

    normalized = uploaded_df.copy()
    normalized.columns = [_normalize_col_name(col) for col in normalized.columns]
    alias_map = {
        "student_id": "student_id_perf",
        "id": "student_id_perf",
        "roll_no": "student_id_perf",
        "rollno": "student_id_perf",
        "cgpa": "mean_cgpa",
        "current_cgpa": "mean_cgpa",
        "sgpa": "mean_sgpa",
        "backlogs": "total_backlogs",
        "backlog": "total_backlogs",
        "attendance": "attendance_rate",
        "attendance_percent": "attendance_rate",
        "stress": "stress_numeric",
        "study_hours": "study_hours_day",
        "placement_readiness": "placement_readiness_score",
        "career_recommendation": "career_path",
    }
    normalized.rename(columns={k: v for k, v in alias_map.items() if k in normalized.columns}, inplace=True)

    if "mean_cgpa" not in normalized.columns and "latest_cgpa" in normalized.columns:
        normalized["mean_cgpa"] = pd.to_numeric(normalized["latest_cgpa"], errors="coerce")
    if "mean_cgpa" not in normalized.columns:
        return False, "Uploaded dataset must include CGPA information (mean_cgpa or cgpa)."

    if "student_id_perf" not in normalized.columns:
        normalized["student_id_perf"] = np.arange(1, len(normalized) + 1)
    if "latest_cgpa" not in normalized.columns:
        normalized["latest_cgpa"] = pd.to_numeric(normalized["mean_cgpa"], errors="coerce")
    if "mean_sgpa" not in normalized.columns:
        normalized["mean_sgpa"] = pd.to_numeric(normalized["mean_cgpa"], errors="coerce")
    if "total_backlogs" not in normalized.columns:
        normalized["total_backlogs"] = 0

    prepared = prepare_student_dataset(normalized)
    subject_summary = _build_subject_summary_from_uploaded(normalized, prepared)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PROCESSED_DIR / "final_student_dataset.csv", index=False)
    subject_summary.to_csv(PROCESSED_DIR / "subject_summary.csv", index=False)

    return (
        True,
        (
            f"Activated dataset with {len(prepared)} students. "
            f"Subject summary rows: {len(subject_summary)}."
        ),
    )


def write_etl_status(
    new_records: int | str = "Unknown",
    errors: str = "None",
    status: str = "Success",
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    run_time = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %I:%M %p")
    payload = {
        "last_run": run_time,
        "new_records": str(new_records),
        "errors": errors if errors else "None",
        "status": status,
    }
    ETL_STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_etl_status() -> dict[str, str]:
    if ETL_STATUS_FILE.exists():
        try:
            payload = json.loads(ETL_STATUS_FILE.read_text(encoding="utf-8"))
            return {
                "last_run": str(payload.get("last_run", "Unavailable")),
                "new_records": str(payload.get("new_records", "Unknown")),
                "errors": str(payload.get("errors", "None")),
                "status": str(payload.get("status", "Unknown")),
            }
        except json.JSONDecodeError:
            pass

    target = PROCESSED_DIR / "final_student_dataset.csv"
    if not target.exists():
        return {
            "last_run": "Unavailable",
            "new_records": "Unknown",
            "errors": "Processed file not found",
            "status": "Failed",
        }
    modified = pd.Timestamp(target.stat().st_mtime, unit="s").tz_localize("UTC")
    local_time = modified.tz_convert("Asia/Kolkata")
    return {
        "last_run": local_time.strftime("%Y-%m-%d %I:%M %p"),
        "new_records": "N/A",
        "errors": "None",
        "status": "Success",
    }


def get_data_layers_status() -> dict[str, dict[str, str]]:
    def _layer(path: Path) -> dict[str, str]:
        if not path.exists():
            return {"files": "0", "last_update": "Unavailable"}
        files = [p for p in path.glob("*.csv") if p.is_file()]
        if not files:
            return {"files": "0", "last_update": "Unavailable"}
        last_file = max(files, key=lambda f: f.stat().st_mtime)
        ts = pd.Timestamp(last_file.stat().st_mtime, unit="s").tz_localize("UTC")
        local = ts.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %I:%M %p")
        return {"files": str(len(files)), "last_update": local}

    return {
        "bronze": _layer(BRONZE_DIR),
        "silver": _layer(SILVER_DIR),
        "gold": _layer(GOLD_DIR),
    }


def optimize_intervention_plan(
    df: pd.DataFrame, budget_hours: float = 40.0, max_students: int = 30
) -> tuple[pd.DataFrame, dict[str, float]]:
    candidates = df[df["risk_level"].isin(["High", "Medium"])].copy()
    if candidates.empty:
        return pd.DataFrame(), {"selected_students": 0, "used_hours": 0.0, "projected_gain": 0.0}

    candidates["mentoring_hours"] = (
        1.2
        + np.clip(candidates["stress_numeric"], 0, 3) * 0.7
        + np.clip((0.85 - candidates["attendance_rate"]) * 4.0, 0, 3)
        + np.clip(candidates["total_backlogs"], 0, 6) * 0.25
    ).clip(1.0, 5.0)

    candidates["projected_cgpa_gain"] = (
        np.clip(7.0 - candidates["mean_cgpa"], 0, 4) * 0.22
        + np.clip((0.80 - candidates["attendance_rate"]) * 1.6, 0, 1.0)
        + np.clip(candidates["stress_numeric"] - 1, 0, 2) * 0.10
    ).clip(0.05, 1.40)

    candidates["efficiency"] = candidates["projected_cgpa_gain"] / candidates["mentoring_hours"]
    ranked = candidates.sort_values(["efficiency", "academic_risk_score"], ascending=[False, False])

    selected_rows: list[pd.Series] = []
    used_hours = 0.0
    for _, row in ranked.iterrows():
        hours = float(row["mentoring_hours"])
        if used_hours + hours > budget_hours:
            continue
        selected_rows.append(row)
        used_hours += hours
        if len(selected_rows) >= max_students:
            break

    if not selected_rows:
        return pd.DataFrame(), {"selected_students": 0, "used_hours": 0.0, "projected_gain": 0.0}

    selected = pd.DataFrame(selected_rows).copy()
    selected["student_id_perf"] = selected["student_id_perf"].astype(int)
    selected = selected[
        [
            "student_id_perf",
            "risk_level",
            "risk_reason",
            "mentoring_hours",
            "projected_cgpa_gain",
            "efficiency",
        ]
    ].rename(
        columns={
            "student_id_perf": "Student",
            "risk_level": "Risk Level",
            "risk_reason": "Reason",
            "mentoring_hours": "Mentoring Hours",
            "projected_cgpa_gain": "Projected CGPA Gain",
            "efficiency": "Impact per Hour",
        }
    )
    selected["Student"] = selected["Student"].apply(lambda sid: f"Student {sid}")
    selected["Mentoring Hours"] = selected["Mentoring Hours"].round(2)
    selected["Projected CGPA Gain"] = selected["Projected CGPA Gain"].round(3)
    selected["Impact per Hour"] = selected["Impact per Hour"].round(3)

    summary = {
        "selected_students": float(len(selected)),
        "used_hours": round(float(selected["Mentoring Hours"].sum()), 2),
        "projected_gain": round(float(selected["Projected CGPA Gain"].sum()), 2),
    }
    return selected, summary


def _build_rag_documents(df: pd.DataFrame, subject_df: pd.DataFrame) -> list[str]:
    docs: list[str] = []

    at_risk_count = int(df["at_risk_student"].sum())
    docs.append(
        (
            f"Institution summary: students={len(df)}, at_risk={at_risk_count}, "
            f"avg_cgpa={df['mean_cgpa'].mean():.2f}, avg_readiness={df['placement_readiness_score'].mean():.1f}%."
        )
    )

    top_subjects = subject_df.sort_values("difficulty_score", ascending=False).head(20)
    for _, row in top_subjects.iterrows():
        docs.append(
            (
                f"Subject {row['subject']} has difficulty score {row['difficulty_score']:.2f}, "
                f"failure rate {(row.get('fail_rate', 0.0) * 100):.1f}% and average theory score "
                f"{(row.get('avg_theory_gp', 0.0) * 10):.1f}."
            )
        )

    student_cols = [
        "student_id_perf",
        "risk_level",
        "mean_cgpa",
        "attendance_rate",
        "stress_numeric",
        "career_path",
        "placement_readiness_score",
        "risk_reason",
        "academic_risk_score",
    ]
    existing_cols = [col for col in student_cols if col in df.columns]
    student_rows = df[existing_cols].copy()
    if "academic_risk_score" in student_rows.columns:
        student_rows = student_rows.sort_values("academic_risk_score", ascending=False)
    student_rows = student_rows.head(350)
    for _, row in student_rows.iterrows():
        docs.append(
            (
                f"Student {int(row['student_id_perf'])}: risk={row['risk_level']}, "
                f"cgpa={row['mean_cgpa']:.2f}, attendance={(row['attendance_rate'] * 100):.1f}%, "
                f"stress={row['stress_numeric']:.0f}, career={row['career_path']}, "
                f"readiness={row['placement_readiness_score']:.1f}%, reason={row['risk_reason']}."
            )
        )
    return docs


def generate_rag_response(question: str, df: pd.DataFrame, subject_df: pd.DataFrame) -> str:
    docs = _build_rag_documents(df, subject_df)
    if not docs:
        return "No indexed context is available for retrieval yet."

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return "Retrieval dependencies are unavailable."

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).ravel()

    top_idx = np.argsort(scores)[::-1][:4]
    top_hits = [(docs[i], float(scores[i])) for i in top_idx if scores[i] > 0]
    if not top_hits:
        return (
            "I could not find strong matches in indexed student data. "
            "Try asking about risk, attendance, stress, subject difficulty, or student ID."
        )

    lines = [f"- {text}" for text, _ in top_hits[:3]]
    return "RAG response from indexed data:\n" + "\n".join(lines)


def generate_advisor_response(question: str, df: pd.DataFrame, subject_df: pd.DataFrame) -> str:
    query = question.strip().lower()
    if not query:
        return "Ask about risk, hardest subjects, placement readiness, or a student ID."

    id_match = re.search(r"\b(\d{1,4})\b", query)
    student_row: pd.Series | None = None
    if id_match and "student_id_perf" in df.columns:
        student_id = int(id_match.group(1))
        match = df[df["student_id_perf"] == student_id]
        if not match.empty:
            student_row = match.iloc[0]

    if "at risk" in query or "risk" in query:
        count = int(df["at_risk_student"].sum())
        top_reasons = (
            df.loc[df["at_risk_student"], "risk_reason"]
            .str.split(",")
            .explode()
            .str.strip()
            .value_counts()
            .head(2)
            .index.tolist()
        )
        reason_text = ", ".join(top_reasons) if top_reasons else "multiple risk factors"
        return f"{count} students are currently at risk, mainly due to {reason_text}."

    if "hardest" in query or "difficult" in query:
        if subject_df.empty:
            return "Subject data is not available right now."
        hardest = subject_df.sort_values("difficulty_score", ascending=False).iloc[0]
        fail_rate = hardest.get("fail_rate", 0.0) * 100
        return f"{hardest['subject']} appears hardest with about {fail_rate:.1f}% failure rate."

    if "placement" in query or "career" in query:
        if student_row is not None:
            gaps = infer_skill_gaps(student_row)
            return (
                f"Student {int(student_row['student_id_perf'])} readiness is "
                f"{student_row['placement_readiness_score']:.1f}%. "
                f"Recommended path: {student_row['career_path']}. "
                f"Skill gaps: {', '.join(gaps[:3])}."
            )
        avg_ready = df["placement_readiness_score"].mean()
        top_path = df["career_path"].value_counts().index[0]
        return (
            f"Institutional placement readiness is {avg_ready:.1f}%. "
            f"Most common recommended path is {top_path}."
        )

    if "internship" in query:
        rate = df["internship_participation"].mean() * 100
        return f"Internship participation is currently {rate:.1f}% of students."

    if "stress" in query:
        stress_impact = (
            df.groupby("stress_numeric")["mean_cgpa"].mean().sort_index(ascending=True).to_dict()
        )
        if 1 in stress_impact and 3 in stress_impact:
            drop = stress_impact[1] - stress_impact[3]
            return f"Students in high stress bands average about {drop:.2f} lower CGPA than low stress peers."
        return "Stress has a visible negative relationship with academic performance."

    rag_result = generate_rag_response(question, df, subject_df)
    return (
        "I can answer risk, placement, subject difficulty, internship, and student-specific "
        "questions. Here is what I found:\n"
        f"{rag_result}"
    )

