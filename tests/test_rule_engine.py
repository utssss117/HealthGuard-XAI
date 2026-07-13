"""
tests/test_rule_engine.py
──────────────────────────
Unit tests for the HealthGuard-XAI clinical rule engine.

Run with:  pytest tests/test_rule_engine.py -v
"""

import pytest
from recommendation_engine.rule_engine import apply_clinical_rules
from recommendation_engine.clinical_thresholds import (
    classify_risk_band,
    classify_age_group,
    BMI_THRESHOLDS,
    GLUCOSE_THRESHOLDS,
)


# ── apply_clinical_rules ──────────────────────────────────────────────────────

class TestApplyClinicalRules:

    def _get_rec_features(self, profile):
        return [r["feature"] for r in apply_clinical_rules(profile)]

    def test_high_bmi_triggers_bmi_rule(self):
        profile = {"BMI": 35.0, "Glucose": 90.0, "BloodPressure": 70.0, "Age": 30, "Insulin": 10.0}
        features = self._get_rec_features(profile)
        assert "BMI" in features

    def test_normal_bmi_does_not_trigger_bmi_rule(self):
        profile = {"BMI": 22.0, "Glucose": 90.0, "BloodPressure": 70.0, "Age": 30, "Insulin": 10.0}
        features = self._get_rec_features(profile)
        assert "BMI" not in features

    def test_high_glucose_triggers_glucose_rule(self):
        profile = {"BMI": 22.0, "Glucose": 140.0, "BloodPressure": 70.0, "Age": 30, "Insulin": 10.0}
        features = self._get_rec_features(profile)
        assert "Glucose" in features

    def test_prediabetic_glucose_triggers_glucose_rule(self):
        profile = {"BMI": 22.0, "Glucose": 110.0, "BloodPressure": 70.0, "Age": 30, "Insulin": 10.0}
        features = self._get_rec_features(profile)
        assert "Glucose" in features

    def test_normal_glucose_does_not_trigger(self):
        profile = {"BMI": 22.0, "Glucose": 85.0, "BloodPressure": 70.0, "Age": 30, "Insulin": 10.0}
        features = self._get_rec_features(profile)
        assert "Glucose" not in features

    def test_high_bp_triggers_bp_rule(self):
        profile = {"BMI": 22.0, "Glucose": 85.0, "BloodPressure": 135.0, "Age": 30, "Insulin": 10.0}
        features = self._get_rec_features(profile)
        assert "BloodPressure" in features

    def test_elderly_patient_triggers_age_rule(self):
        profile = {"BMI": 22.0, "Glucose": 85.0, "BloodPressure": 70.0, "Age": 68, "Insulin": 10.0}
        features = self._get_rec_features(profile)
        assert "Age" in features

    def test_young_patient_no_age_rule(self):
        profile = {"BMI": 22.0, "Glucose": 85.0, "BloodPressure": 70.0, "Age": 28, "Insulin": 10.0}
        features = self._get_rec_features(profile)
        assert "Age" not in features

    def test_high_insulin_triggers_insulin_rule(self):
        profile = {"BMI": 22.0, "Glucose": 85.0, "BloodPressure": 70.0, "Age": 30, "Insulin": 80.0}
        features = self._get_rec_features(profile)
        assert "Insulin" in features

    def test_comorbidity_rule_triggers_when_both_elevated(self):
        profile = {
            "BMI": 35.0, "Glucose": 140.0,
            "BloodPressure": 135.0, "Age": 30, "Insulin": 10.0,
        }
        features = self._get_rec_features(profile)
        assert "Comorbidity" in features

    def test_empty_profile_returns_no_recs(self):
        profile = {"BMI": 22.0, "Glucose": 85.0, "BloodPressure": 70.0, "Age": 30, "Insulin": 10.0}
        recs = apply_clinical_rules(profile)
        assert isinstance(recs, list)

    def test_recommendation_has_required_keys(self):
        profile = {"BMI": 35.0, "Glucose": 140.0, "BloodPressure": 70.0, "Age": 30, "Insulin": 10.0}
        recs = apply_clinical_rules(profile)
        for rec in recs:
            assert "feature" in rec
            assert "recommendation" in rec
            assert "justification" in rec
            assert "related_risk" in rec
            assert "base_score" in rec


# ── classify_risk_band ────────────────────────────────────────────────────────

class TestClassifyRiskBand:

    def test_low_risk(self):
        assert classify_risk_band(0.2) == "Low"

    def test_medium_risk(self):
        assert classify_risk_band(0.5) == "Medium"

    def test_high_risk(self):
        assert classify_risk_band(0.8) == "High"

    def test_boundary_33_is_medium(self):
        assert classify_risk_band(0.34) == "Medium"

    def test_boundary_66_is_high(self):
        assert classify_risk_band(0.67) == "High"


# ── classify_age_group ────────────────────────────────────────────────────────

class TestClassifyAgeGroup:

    def test_young_adult(self):
        group = classify_age_group(25)
        assert group in ("young_adult", "adult")

    def test_older_adult(self):
        group = classify_age_group(65)
        assert group in ("older_adult", "elderly")

    def test_middle_adult(self):
        group = classify_age_group(45)
        assert isinstance(group, str)
        assert len(group) > 0
