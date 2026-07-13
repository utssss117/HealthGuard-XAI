"""
tests/test_safety_filter.py
────────────────────────────
Unit tests for the HealthGuard-XAI safety filter / guardrail engine.

Run with:  pytest tests/test_safety_filter.py -v
"""

import pytest
from health_llm_assistant.safety_filter import check_safety, SafetyResult


# ── Emergency Detection ───────────────────────────────────────────────────────

class TestEmergencyDetection:

    def test_chest_pain_triggers_emergency(self):
        result = check_safety("I have severe chest pain right now")
        assert result.safety_flag is True
        assert result.escalation_required is True
        assert result.triggered_layer == "emergency"
        assert result.is_safe is False

    def test_heart_attack_triggers_emergency(self):
        result = check_safety("I think I'm having a heart attack")
        assert result.triggered_layer == "emergency"
        assert result.escalation_required is True

    def test_stroke_triggers_emergency(self):
        result = check_safety("my face is drooping and I have slurred speech")
        assert result.triggered_layer == "emergency"

    def test_suicide_triggers_emergency(self):
        result = check_safety("I'm having suicidal thoughts")
        assert result.triggered_layer == "emergency"
        assert result.escalation_required is True

    def test_emergency_response_contains_crisis_number(self):
        result = check_safety("chest pain")
        assert "911" in result.redirection_message or "emergency" in result.redirection_message.lower()


# ── Medication Boundary ───────────────────────────────────────────────────────

class TestMedicationBoundary:

    def test_dosage_request_triggers_medication(self):
        result = check_safety("what dosage of metformin should I take?")
        assert result.safety_flag is True
        assert result.triggered_layer == "medication"
        assert result.escalation_required is False

    def test_prescription_triggers_medication(self):
        result = check_safety("can you prescribe me something for my blood sugar?")
        assert result.triggered_layer == "medication"

    def test_stop_medication_triggers_boundary(self):
        result = check_safety("should I stop taking my medication?")
        assert result.safety_flag is True

    def test_medication_response_recommends_physician(self):
        result = check_safety("how much insulin should I take")
        assert "physician" in result.redirection_message.lower() or "pharmacist" in result.redirection_message.lower()


# ── Diagnostic Boundary ───────────────────────────────────────────────────────

class TestDiagnosticBoundary:

    def test_do_i_have_triggers_diagnostic(self):
        result = check_safety("do I have diabetes?")
        assert result.safety_flag is True
        assert result.triggered_layer == "diagnostic"

    def test_am_i_diabetic_triggers_diagnostic(self):
        result = check_safety("am I diabetic based on my numbers?")
        assert result.triggered_layer == "diagnostic"

    def test_diagnose_me_triggers_diagnostic(self):
        result = check_safety("please diagnose my condition")
        assert result.triggered_layer == "diagnostic"

    def test_diagnostic_response_contains_disclaimer(self):
        result = check_safety("do I have heart disease?")
        assert "clinical diagnosis" in result.redirection_message.lower() or "diagnos" in result.redirection_message.lower()


# ── Safe Inputs ───────────────────────────────────────────────────────────────

class TestSafeInputs:

    def test_lifestyle_question_is_safe(self):
        result = check_safety("what foods should I eat to lower my blood sugar?")
        assert result.is_safe is True
        assert result.safety_flag is False
        assert result.escalation_required is False
        assert result.redirection_message is None

    def test_exercise_question_is_safe(self):
        result = check_safety("how much exercise do I need per week?")
        assert result.is_safe is True

    def test_explain_risk_is_safe(self):
        result = check_safety("can you explain what my glucose level means?")
        assert result.is_safe is True

    def test_empty_string_is_safe(self):
        result = check_safety("")
        assert result.is_safe is True

    def test_general_health_question_is_safe(self):
        result = check_safety("what is HbA1c?")
        assert result.is_safe is True


# ── Priority Ordering ─────────────────────────────────────────────────────────

class TestPriorityOrdering:

    def test_emergency_takes_precedence_over_medication(self):
        """If emergency + medication are both present, emergency layer wins."""
        result = check_safety("I'm having chest pain and need to know my metformin dosage")
        assert result.triggered_layer == "emergency"

    def test_medication_takes_precedence_over_diagnostic(self):
        """If medication + diagnostic are both present, medication layer wins."""
        result = check_safety("do I have diabetes and should I prescribe insulin?")
        assert result.triggered_layer == "medication"
