"""
data_cleaning.py
Cleans each dataset individually: nulls, duplicates, column normalisation.
"""
import pandas as pd
import numpy as np
import re


# ── helpers ──────────────────────────────────────────────────────────────────

def _norm_col(name: str) -> str:
    """Lowercase, strip, replace spaces/special chars with underscore."""
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    return name.strip('_')


def _norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_norm_col(c) for c in df.columns]
    return df


# ── per-dataset cleaners ──────────────────────────────────────────────────────

def clean_performance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _norm_columns(df)
    df.drop_duplicates(inplace=True)

    # Fill numeric nulls with sensible defaults
    for col in ['prsgpa', 'prcgpa', 'noofbacklog']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in ['theoryaggggradepoint', 'practicalagggradepoint',
                'theoryagggradepoint', 'practicalagggradepoint',
                'thispresent', 'prispresent']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ['sgpa', 'cgpa']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Normalise grade strings
    for col in ['theoryagggrade', 'practicalagggrade']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.upper().fillna('NA')

    df.reset_index(drop=True, inplace=True)
    print(f"[CLEAN] performance -> {df.shape}")
    return df


def clean_attitude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _norm_columns(df)
    df.drop_duplicates(inplace=True)

    # Add synthetic student_id (row-based, 1-indexed)
    df.insert(0, 'student_id', range(1, len(df) + 1))

    # Numeric fill
    for col in ['height_cm', 'weight_kg', '10th_mark', '12th_mark', 'college_mark',
                'salary_expectation']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical fill
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Strip trailing spaces from all string columns
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    df.reset_index(drop=True, inplace=True)
    print(f"[CLEAN] attitude/behaviour -> {df.shape}")
    return df


def clean_research(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _norm_columns(df)
    df.drop_duplicates(inplace=True)

    # Drop rows where ALL values are null (header artefacts)
    df.dropna(how='all', inplace=True)
    df.dropna(subset=['cgpa'], inplace=True)

    # Add synthetic student_id
    df.insert(0, 'student_id_research', range(1, len(df) + 1))

    # Numeric fill
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    df.reset_index(drop=True, inplace=True)
    print(f"[CLEAN] research -> {df.shape}")
    return df


def clean_all(datasets: dict) -> dict:
    cleaned = {}
    cleaned['performance'] = clean_performance(datasets['performance'])
    # Use Student_Attitude_and_Behavior as primary; Student_Behaviour as validation
    att = clean_attitude(datasets['attitude'])
    beh = clean_attitude(datasets['behaviour'])
    # Average numeric columns where files differ
    num_cols = att.select_dtypes(include=[np.number]).columns.difference(['student_id'])
    for col in num_cols:
        att[col] = (att[col] + beh[col]) / 2
    cleaned['lifestyle'] = att
    cleaned['research'] = clean_research(datasets['research'])
    return cleaned


if __name__ == '__main__':
    from data_loader import load_all
    raw = load_all()
    clean = clean_all(raw)
    for k, v in clean.items():
        print(k, v.shape, v.columns.tolist())
