import pandas as pd
import os
from sklearn.datasets import load_breast_cancer
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        # Fallback to structured medical dataset
        print(f"Dataset not found at {filepath}, loading sample Breast Cancer dataset.")
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        return df

def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Fallback if target column name is not found
    if target_col not in df.columns:
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
