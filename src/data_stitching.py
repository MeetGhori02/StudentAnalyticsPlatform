"""
data_stitching.py
Merges all cleaned datasets into a single unified dataset.

Strategy:
  - performance: 514 unique students × 8 semesters. Pivot to one row per student
    with aggregated metrics (mean CGPA, total backlogs, per-semester SGPA, etc.)
  - lifestyle (attitude/behaviour): 235 rows, no explicit student_id, row-index used.
    Map proportionally to performance students via synthetic id.
  - research: 220 engineering students. Merge on normalised 10th / 12th marks +
    gender as approximate key; fall-back to concatenation.
"""
import pandas as pd
import numpy as np
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


# ── helpers ──────────────────────────────────────────────────────────────────

def _pivot_performance(perf: pd.DataFrame) -> pd.DataFrame:
    """One row per student with aggregate academic metrics."""
    grp = perf.groupby('rollno')

    agg = grp.agg(
        latest_cgpa        = ('cgpa',        'last'),
        mean_cgpa          = ('cgpa',        'mean'),
        min_cgpa           = ('cgpa',        'min'),
        max_cgpa           = ('cgpa',        'max'),
        mean_sgpa          = ('sgpa',        'mean'),
        total_backlogs     = ('noofbacklog', 'sum'),
        semesters_attended = ('semester',    'nunique'),
        theory_gp_mean     = ('theoryagggradepoint', 'mean'),
        practical_marks_mean = ('practicalaggmarks', 'mean'),
        th_attendance_rate = ('thispresent', 'mean'),
        pr_attendance_rate = ('prispresent', 'mean'),
    ).reset_index()

    # Pivot SGPA per semester
    sgpa_pivot = (
        perf.drop_duplicates(['rollno', 'semester'])
            .pivot_table(index='rollno', columns='semester', values='sgpa', aggfunc='mean')
    )
    sgpa_pivot.columns = [f'sgpa_sem_{int(c)}' for c in sgpa_pivot.columns]
    sgpa_pivot.reset_index(inplace=True)

    agg = agg.merge(sgpa_pivot, on='rollno', how='left')
    agg.rename(columns={'rollno': 'student_id_perf'}, inplace=True)
    print(f"[STITCH] pivoted performance -> {agg.shape}")
    return agg


def _subject_summary(perf: pd.DataFrame) -> pd.DataFrame:
    """Subject-level difficulty / performance table (separate, for dashboard)."""
    subj = perf.groupby('subjectname').agg(
        avg_theory_gp   = ('theoryagggradepoint', 'mean'),
        avg_practical   = ('practicalaggmarks',   'mean'),
        student_count   = ('rollno',              'nunique'),
        fail_rate       = ('theoryagggrade',       lambda x: (x.isin(['F','AB'])).mean()),
    ).reset_index()
    subj.rename(columns={'subjectname': 'subject'}, inplace=True)
    subj['difficulty_score'] = (1 - subj['avg_theory_gp'] / 10) * 50 + subj['fail_rate'] * 50
    subj.sort_values('difficulty_score', ascending=False, inplace=True)
    return subj


def stitch(cleaned: dict) -> tuple:
    """
    Returns (final_df, subject_df).
    final_df  → one row per student, all features merged.
    subject_df → subject-level analytics.
    """
    perf     = cleaned['performance']
    lifestyle = cleaned['lifestyle']
    research  = cleaned['research']

    # ── 1. Pivot performance ──────────────────────────────────────────────
    perf_pivot = _pivot_performance(perf)
    subj_df    = _subject_summary(perf)

    # ── 2. Attach lifestyle via modulo mapping ────────────────────────────
    # No common key exists; distribute 235 lifestyle rows across 514 perf students
    n_life = len(lifestyle)
    lifestyle_ext = lifestyle.iloc[perf_pivot.index % n_life].reset_index(drop=True)
    # Drop student_id from lifestyle (keep perf id as master)
    lifestyle_ext = lifestyle_ext.drop(columns=['student_id'], errors='ignore')

    combined = pd.concat([perf_pivot.reset_index(drop=True),
                          lifestyle_ext.reset_index(drop=True)], axis=1)

    # ── 3. Attach research via 10th/12th marks fuzzy join ────────────────
    # Normalise marks columns
    research_sub = research[['student_id_research', 'marks_10th', 'marks_12th',
                              'gender', 'branch', 'current_back', 'ever_back',
                              'olympiads_qualified', 'technical_projects',
                              'tech_quiz', 'engg_coaching', 'ntse_scholarships',
                              'miscellany_tech_events']].copy()

    # Round marks to nearest integer for fuzzy key
    research_sub['key_10th'] = research_sub['marks_10th'].round(0)
    research_sub['key_12th'] = research_sub['marks_12th'].round(0)

    if '10th_mark' in combined.columns and '12th_mark' in combined.columns:
        combined['key_10th'] = combined['10th_mark'].round(0)
        combined['key_12th'] = combined['12th_mark'].round(0)
        combined = combined.merge(
            research_sub.drop(columns=['marks_10th', 'marks_12th']),
            on=['key_10th', 'key_12th'],
            how='left'
        )
        combined.drop(columns=['key_10th', 'key_12th'], inplace=True)
    else:
        # Fall-back: modulo mapping
        res_ext = research_sub.iloc[combined.index % len(research_sub)].reset_index(drop=True)
        combined = pd.concat([combined, res_ext.drop(columns=['student_id_research'], errors='ignore')], axis=1)

    combined.drop_duplicates(subset=['student_id_perf'], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"[STITCH] final combined dataset -> {combined.shape}")
    return combined, subj_df


def save(df: pd.DataFrame, subj_df: pd.DataFrame):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = os.path.join(PROCESSED_DIR, 'final_student_dataset.csv')
    df.to_csv(out, index=False)
    print(f"[SAVED] {out}")

    subj_out = os.path.join(PROCESSED_DIR, 'subject_summary.csv')
    subj_df.to_csv(subj_out, index=False)
    print(f"[SAVED] {subj_out}")


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader  import load_all
    from data_cleaning import clean_all
    raw     = load_all()
    cleaned = clean_all(raw)
    final, subjects = stitch(cleaned)
    save(final, subjects)
    print(final.head(2).to_string())
