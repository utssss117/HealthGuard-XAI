"""
tests/test_preprocessing.py
─────────────────────────────
Unit tests for the data preprocessing pipeline.

Run with:  pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from risk_modeling.preprocessing import preprocess_data
from risk_modeling.data_loader import load_data, split_features_target


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small synthetic diabetes-like dataset for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Pregnancies":              np.random.randint(0, 10, n).astype(float),
        "Glucose":                  np.random.uniform(70, 200, n),
        "BloodPressure":            np.random.uniform(50, 130, n),
        "SkinThickness":            np.random.uniform(5, 60, n),
        "Insulin":                  np.random.uniform(0, 400, n),
        "BMI":                      np.random.uniform(18, 45, n),
        "DiabetesPedigreeFunction": np.random.uniform(0.08, 2.5, n),
        "Age":                      np.random.randint(18, 80, n).astype(float),
        "Outcome":                  np.random.randint(0, 2, n),
    })


@pytest.fixture
def features_and_target(sample_df):
    return split_features_target(sample_df, "Outcome")


# ── preprocess_data ───────────────────────────────────────────────────────────

class TestPreprocessData:

    def test_returns_five_outputs(self, features_and_target):
        X, y = features_and_target
        result = preprocess_data(X, y)
        assert len(result) == 5

    def test_train_test_shapes_correct(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(X)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_scaler_is_standard_scaler(self, features_and_target):
        X, y = features_and_target
        _, _, _, _, scaler = preprocess_data(X, y)
        assert isinstance(scaler, StandardScaler)

    def test_train_features_are_scaled(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, _, _, _ = preprocess_data(X, y)
        # Scaled data should have approximately zero mean on training set
        means = X_train.mean()
        assert all(abs(m) < 0.1 for m in means), "Training features not properly scaled"

    def test_no_nans_after_preprocessing(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, y_train, y_test, _ = preprocess_data(X, y)
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()

    def test_handles_missing_values(self, sample_df):
        """Imputer should handle NaN values in input data."""
        df = sample_df.copy()
        df.loc[0:5, "Glucose"] = np.nan
        df.loc[10:15, "BMI"]   = np.nan
        X, y = split_features_target(df, "Outcome")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()

    def test_custom_test_size(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, _, _, _ = preprocess_data(X, y, test_size=0.3)
        expected_test = int(0.3 * len(X))
        # Allow ±2 for rounding
        assert abs(len(X_test) - expected_test) <= 2

    def test_column_names_preserved(self, features_and_target):
        X, y = features_and_target
        X_train, _, _, _, _ = preprocess_data(X, y)
        assert list(X_train.columns) == list(X.columns)

    def test_stratified_split_preserves_class_balance(self, features_and_target):
        X, y = features_and_target
        _, _, y_train, y_test, _ = preprocess_data(X, y, test_size=0.2)
        orig_rate  = y.mean()
        train_rate = y_train.mean()
        test_rate  = y_test.mean()
        # Allow 15% deviation from the original class rate
        assert abs(train_rate - orig_rate) < 0.15
        assert abs(test_rate  - orig_rate) < 0.15


# ── split_features_target ─────────────────────────────────────────────────────

class TestSplitFeaturesTarget:

    def test_correct_split(self, sample_df):
        X, y = split_features_target(sample_df, "Outcome")
        assert "Outcome" not in X.columns
        assert y.name == "Outcome"

    def test_shape_consistent(self, sample_df):
        X, y = split_features_target(sample_df, "Outcome")
        assert len(X) == len(y)

    def test_fallback_on_missing_column(self, sample_df):
        """Should fall back to last column if target not found."""
        X, y = split_features_target(sample_df, "NonExistentTarget")
        assert len(X.columns) == len(sample_df.columns) - 1


# ── Session Tracker ───────────────────────────────────────────────────────────

class TestSessionTracker:

    def test_initial_state(self):
        from utils.session_tracker import SessionTracker
        tracker = SessionTracker()
        assert tracker.count() == 0
        assert tracker.latest() is None
        assert tracker.get_delta() is None

    def test_add_prediction_increments_count(self):
        from utils.session_tracker import SessionTracker
        tracker = SessionTracker()
        biomarkers = {"Glucose": 120.0, "BMI": 28.0, "BloodPressure": 72.0,
                      "Insulin": 80.0, "Age": 35.0}
        tracker.add_prediction(biomarkers, 0.45, "Medium")
        assert tracker.count() == 1

    def test_delta_returns_risk_change(self):
        from utils.session_tracker import SessionTracker
        tracker = SessionTracker()
        b = {"Glucose": 120.0, "BMI": 28.0, "BloodPressure": 72.0, "Insulin": 80.0, "Age": 35.0}
        tracker.add_prediction(b, 0.65, "Medium")
        tracker.add_prediction(b, 0.45, "Medium")
        delta = tracker.get_delta()
        assert delta is not None
        assert abs(delta - (-20.0)) < 0.01

    def test_to_dataframe_not_empty(self):
        from utils.session_tracker import SessionTracker
        tracker = SessionTracker()
        b = {"Glucose": 100.0, "BMI": 22.0, "BloodPressure": 70.0, "Insulin": 50.0, "Age": 30.0}
        tracker.add_prediction(b, 0.25, "Low")
        df = tracker.to_dataframe()
        assert len(df) == 1
        assert "Risk (%)" in df.columns
