"""
ml_models.py
Trains multiple ML models for:
  - CGPA regression (Linear Regression, Random Forest)
  - Risk classification (Logistic Regression, Decision Tree, Random Forest)
Returns trained models + evaluation metrics dict.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# ── feature selection ─────────────────────────────────────────────────────────

REGRESSION_FEATURES = [
    'mean_sgpa', 'total_backlogs', 'th_attendance_rate', 'pr_attendance_rate',
    'theory_gp_mean', 'practical_marks_mean', 'study_hours_day',
    'stress_numeric', 'travel_mins', 'social_hours',
    'academic_risk_score', 'average_marks', 'semesters_attended',
]

CLASSIFICATION_FEATURES = REGRESSION_FEATURES + ['mean_cgpa']


def _select(df: pd.DataFrame, features: list) -> pd.DataFrame:
    available = [f for f in features if f in df.columns]
    return df[available].fillna(df[available].median())


# ── regression: predict CGPA ─────────────────────────────────────────────────

def train_cgpa_models(df: pd.DataFrame) -> dict:
    if 'mean_cgpa' not in df.columns:
        return {}

    X = _select(df, REGRESSION_FEATURES)
    y = df['mean_cgpa'].fillna(df['mean_cgpa'].median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        'Linear Regression':      LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        results[name] = {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'R2': round(r2, 4)}
        trained[name] = model
        print(f"  [{name}] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    return {'models': trained, 'metrics': results,
            'X_test': X_test, 'y_test': y_test, 'features': list(X.columns)}


# ── classification: predict at-risk ──────────────────────────────────────────

def train_risk_models(df: pd.DataFrame) -> dict:
    if 'at_risk_student' not in df.columns:
        return {}

    X = _select(df, CLASSIFICATION_FEATURES)
    y = df['at_risk_student'].astype(int)

    if y.nunique() < 2:
        print("  [SKIP] Only one class present in at_risk_student.")
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'Logistic Regression':       LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree Classifier':  DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest Classifier':  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    trained = {}
    cms = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred, output_dict=True)
        cm  = confusion_matrix(y_test, y_pred)
        results[name] = {
            'Accuracy': round(acc, 4),
            'Precision': round(rep.get('1', rep.get('True', {})).get('precision', 0), 4),
            'Recall':    round(rep.get('1', rep.get('True', {})).get('recall', 0), 4),
            'F1':        round(rep.get('1', rep.get('True', {})).get('f1-score', 0), 4),
        }
        trained[name] = model
        cms[name] = cm
        print(f"  [{name}] Acc={acc:.4f}")

    # Feature importance from RF
    rf = trained.get('Random Forest Classifier')
    feat_imp = {}
    if rf and hasattr(rf, 'feature_importances_'):
        feat_imp = dict(zip(X.columns, rf.feature_importances_.round(4)))

    return {'models': trained, 'metrics': results, 'confusion_matrices': cms,
            'feature_importance': feat_imp, 'X_test': X_test, 'y_test': y_test,
            'features': list(X.columns)}


def run_all_models(df: pd.DataFrame) -> dict:
    print("\n[ML] Training CGPA Regression Models...")
    reg_results = train_cgpa_models(df)

    print("\n[ML] Training Risk Classification Models...")
    clf_results = train_risk_models(df)

    return {'regression': reg_results, 'classification': clf_results}


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'final_student_dataset.csv'))
    results = run_all_models(df)
    print("\n== Regression Metrics ==")
    for k, v in results['regression'].get('metrics', {}).items():
        print(f"  {k}: {v}")
    print("\n== Classification Metrics ==")
    for k, v in results['classification'].get('metrics', {}).items():
        print(f"  {k}: {v}")
