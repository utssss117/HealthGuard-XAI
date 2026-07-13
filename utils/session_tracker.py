"""
session_tracker.py
──────────────────
In-session prediction history tracker for HealthGuard-XAI.

Stores all risk assessments made during the current Streamlit session,
enabling trend analysis and before/after comparisons.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd


class SessionTracker:
    """
    Lightweight in-memory store for prediction history within a single session.
    Designed to be stored in st.session_state.
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def add_prediction(
        self,
        biomarkers: Dict[str, float],
        risk_probability: float,
        risk_level: str,
        label: Optional[str] = None,
    ) -> None:
        """
        Record a new prediction.

        Parameters
        ----------
        biomarkers       : Raw patient biomarker dict
        risk_probability : Predicted probability (0–1)
        risk_level       : "Low" | "Medium" | "High"
        label            : Optional user-defined label for this assessment
        """
        n = len(self._records) + 1
        self._records.append(
            {
                "Assessment #":   n,
                "Timestamp":      datetime.now().strftime("%H:%M:%S"),
                "Label":          label or f"Assessment {n}",
                "Risk (%)":       round(risk_probability * 100, 1),
                "Risk Level":     risk_level,
                "Glucose":        biomarkers.get("Glucose", 0),
                "BMI":            biomarkers.get("BMI", 0),
                "BloodPressure":  biomarkers.get("BloodPressure", 0),
                "Insulin":        biomarkers.get("Insulin", 0),
                "Age":            biomarkers.get("Age", 0),
                "_biomarkers":    biomarkers,  # full dict for comparison
            }
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all recorded predictions."""
        return self._records.copy()

    def to_dataframe(self) -> pd.DataFrame:
        """Return history as a display-ready DataFrame (excludes internal fields)."""
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        display_cols = [
            "Assessment #", "Timestamp", "Label",
            "Risk (%)", "Risk Level", "Glucose", "BMI",
            "BloodPressure", "Insulin", "Age",
        ]
        return df[[c for c in display_cols if c in df.columns]]

    def get_risk_trend(self) -> Dict[str, List]:
        """Return data suitable for a Plotly line chart."""
        if not self._records:
            return {"labels": [], "risks": [], "levels": []}
        return {
            "labels": [r["Label"] for r in self._records],
            "risks":  [r["Risk (%)"] for r in self._records],
            "levels": [r["Risk Level"] for r in self._records],
        }

    def get_delta(self) -> Optional[float]:
        """
        Return the change in risk (%) between the most recent and first assessment.
        Returns None if fewer than 2 assessments exist.
        """
        if len(self._records) < 2:
            return None
        return self._records[-1]["Risk (%)"] - self._records[0]["Risk (%)"]

    def count(self) -> int:
        return len(self._records)

    def latest(self) -> Optional[Dict[str, Any]]:
        return self._records[-1] if self._records else None

    def clear(self) -> None:
        self._records.clear()
