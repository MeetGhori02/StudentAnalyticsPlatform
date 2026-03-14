"""
feature_engineering.py
Derives new analytical features and encodes categoricals for ML.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# ── study hours mapping ───────────────────────────────────────────────────────
STUDY_MAP = {
    '0 - 30 minute':   0.25,
    '30 - 60 minute':  0.75,
    '1 - 2 Hour':      1.5,
    '2 - 3 hour':      2.5,
    '3 - 4 hour':      3.5,
    'More Than 4 hour':5.0,
}

STRESS_MAP = {
    'Good':   1,
    'Bad':    2,
    'Awful':  3,
    'Fabulous': 0,
}

TRAVEL_MAP = {
    '0 - 30 minutes': 15,
    '30 - 60 minutes': 45,
    '1 - 1.30 hour': 75,
    '1.30 - 2 hour': 105,
    'More than 2 hour': 150,
}

SOCIAL_MAP = {
    '0 - 30 minutes': 0.25,
    '30 - 60 minutes': 0.75,
    '1 - 1.30 hour': 1.25,
    '1.30 - 2 hour': 1.75,
    'More than 2 hour': 2.5,
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── numeric conversions ──────────────────────────────────────────────
    if 'daily_studing_time' in df.columns:
        df['study_hours_day'] = df['daily_studing_time'].map(STUDY_MAP).fillna(1.0)

    if 'stress_level' in df.columns:
        df['stress_numeric'] = df['stress_level'].map(STRESS_MAP).fillna(1)

    if 'travelling_time' in df.columns:
        df['travel_mins'] = df['travelling_time'].map(TRAVEL_MAP).fillna(45)

    if 'social_medai_video' in df.columns:
        df['social_hours'] = df['social_medai_video'].map(SOCIAL_MAP).fillna(1.0)

    # ── derived academic features ─────────────────────────────────────────
    mark_cols = []
    for col in ['10th_mark', '12th_mark', 'college_mark']:
        if col in df.columns:
            mark_cols.append(col)

    if mark_cols:
        df['average_marks'] = df[mark_cols].mean(axis=1)
    elif 'mean_cgpa' in df.columns:
        df['average_marks'] = df['mean_cgpa'] * 10

    # academic_risk_score: weighted composite
    risk = pd.Series(0.0, index=df.index)
    if 'mean_cgpa' in df.columns:
        cgpa_risk = np.clip((7 - df['mean_cgpa']) / 7, 0, 1) * 40
        risk += cgpa_risk
    if 'total_backlogs' in df.columns:
        bl_risk = np.clip(df['total_backlogs'] / 10, 0, 1) * 30
        risk += bl_risk
    if 'stress_numeric' in df.columns:
        risk += (df['stress_numeric'] / 3) * 15
    if 'th_attendance_rate' in df.columns:
        att_risk = np.clip((0.75 - df['th_attendance_rate']), 0, 0.75) / 0.75 * 15
        risk += att_risk
    df['academic_risk_score'] = risk.round(2)

    # study_efficiency = output (cgpa) / input (study hours)
    if 'mean_cgpa' in df.columns and 'study_hours_day' in df.columns:
        df['study_efficiency'] = (df['mean_cgpa'] / df['study_hours_day']).round(2)
        df['study_efficiency'] = df['study_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # ── at_risk_student flag ─────────────────────────────────────────────
    risk_flag = pd.Series(False, index=df.index)
    if 'mean_cgpa'       in df.columns: risk_flag |= df['mean_cgpa'] < 5
    if 'total_backlogs'  in df.columns: risk_flag |= df['total_backlogs'] > 2
    if 'th_attendance_rate' in df.columns: risk_flag |= df['th_attendance_rate'] < 0.6
    if 'stress_numeric'  in df.columns and 'study_hours_day' in df.columns:
        risk_flag |= (df['stress_numeric'] >= 2) & (df['study_hours_day'] < 1)
    df['at_risk_student'] = risk_flag

    # ── CGPA band for classification target ──────────────────────────────
    if 'mean_cgpa' in df.columns:
        df['cgpa_band'] = pd.cut(
            df['mean_cgpa'],
            bins=[0, 4, 5.5, 7, 8.5, 10],
            labels=['Fail', 'Poor', 'Average', 'Good', 'Excellent']
        )

    print(f"[FEAT] engineered -> {df.shape}  at_risk: {df['at_risk_student'].sum()}")
    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Returns (df_encoded, label_encoders, scaler, feature_cols).
    """
    df = df.copy()

    cat_cols = [c for c in df.select_dtypes(include='object').columns
                if c not in ['student_id_perf', 'subjectname', 'subject']]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # ── scale numeric features ────────────────────────────────────────────
    skip = {'at_risk_student'}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in skip and 'student_id' not in c and '_enc' not in c]

    scaler = MinMaxScaler()
    df_scaled = df.copy()
    if num_cols:
        df_scaled[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))

    return df_scaled, encoders, scaler, num_cols


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    path = os.path.join(PROCESSED_DIR, 'final_student_dataset.csv')
    df = pd.read_csv(path)
    df = engineer(df)
    df.to_csv(os.path.join(PROCESSED_DIR, 'final_student_dataset.csv'), index=False)
    print(df[['mean_cgpa','academic_risk_score','study_efficiency','at_risk_student']].head())
