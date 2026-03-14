"""
data_loader.py
Loads all four raw datasets and returns them as DataFrames.
"""
import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


def load_all():
    """Return a dict of {name: DataFrame} for every raw source."""
    paths = {
        'performance':  os.path.join(RAW_DIR, 'Student_Performance.xlsx'),
        'attitude':     os.path.join(RAW_DIR, 'Student_Attitude_and_Behavior.csv'),
        'behaviour':    os.path.join(RAW_DIR, 'Student_Behaviour.csv'),
        'research':     os.path.join(RAW_DIR, 'research_student__1_.xlsx'),
    }
    dfs = {}
    for name, path in paths.items():
        if path.endswith('.xlsx'):
            dfs[name] = pd.read_excel(path)
        else:
            dfs[name] = pd.read_csv(path)
        print(f"[LOADED] {name}: {dfs[name].shape}")
    return dfs


if __name__ == '__main__':
    datasets = load_all()
    for k, v in datasets.items():
        print(k, v.shape)
