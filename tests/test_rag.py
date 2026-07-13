"""
test_rag.py
───────────
Unit and integration tests for the clinical guidelines Retrieval-Augmented Generation (RAG) feature.
"""

import pytest
from health_llm_assistant.rag_retriever import ClinicalRAGRetriever
from health_llm_assistant.assistant import ask

def test_retriever_initialization():
    """Verify that retriever loads and indexes guidelines database correctly."""
    retriever = ClinicalRAGRetriever()
    assert len(retriever.guidelines) > 0
    assert retriever.tfidf_matrix is not None
    assert "content" in retriever.guidelines[0]
    assert "title" in retriever.guidelines[0]

def test_retrieval_glucose_query():
    """Verify that a glucose-related query retrieves diabetes guidelines."""
    retriever = ClinicalRAGRetriever()
    results = retriever.retrieve("What is a normal fasting glucose level?", top_k=2)
    
    assert len(results) > 0
    # The top result should be either diabetes diagnostics or diabetes diet management
    top_ids = [doc["id"] for doc in results]
    assert any(x in top_ids for x in ["diabetes_diagnostics", "diabetes_dietary_management"])
    assert results[0]["score"] > 0.0

def test_retrieval_blood_pressure_query():
    """Verify that a blood pressure query retrieves blood pressure categories."""
    retriever = ClinicalRAGRetriever()
    results = retriever.retrieve("My blood pressure is 135/85, is that high?", top_k=2)
    
    assert len(results) > 0
    top_ids = [doc["id"] for doc in results]
    assert "blood_pressure_categories" in top_ids
    assert results[0]["score"] > 0.0

def test_retrieval_empty_query():
    """Verify that empty queries return no documents."""
    retriever = ClinicalRAGRetriever()
    results = retriever.retrieve("", top_k=2)
    assert len(results) == 0

def test_format_context():
    """Verify context formatting is structured as expected."""
    retriever = ClinicalRAGRetriever()
    mock_docs = [
        {"title": "Guideline 1", "content": "Text 1", "score": 0.8},
        {"title": "Guideline 2", "content": "Text 2", "score": 0.6}
    ]
    formatted = retriever.format_context_for_prompt(mock_docs)
    assert "Source [1]: Guideline 1" in formatted
    assert "Guideline: Text 1" in formatted
    assert "Source [2]: Guideline 2" in formatted
    assert "Guideline: Text 2" in formatted

def test_empty_format_context():
    """Verify empty matches format gracefully."""
    retriever = ClinicalRAGRetriever()
    formatted = retriever.format_context_for_prompt([])
    assert "No specific official guidelines matches found" in formatted

def test_assistant_rag_integration(monkeypatch):
    """Verify that calling ask retrieves context and passes it to the LLM call."""
    # Mock LLM call to return a stub response, avoiding live API requests
    def mock_call_llm(system_prompt, messages, **kwargs):
        # Assert that the system prompt includes clinical reference indicators
        assert "CLINICAL GUIDELINES REFERENCE" in system_prompt
        assert "Diabetes" in system_prompt or "Glucose" in system_prompt
        return "Mocked guidance response."

    monkeypatch.setattr("health_llm_assistant.assistant.call_llm", mock_call_llm)

    patient_ctx = {
        "patient_profile": {
            "age": 45,
            "bmi": 28.5,
            "blood_pressure": 130,
            "cholesterol": 220,
            "glucose": 110,
        },
        "predicted_risks": {"diabetes": 0.45},
        "risk_level": "Medium",
        "top_risk_factors": ["Fasting Glucose", "BMI"],
        "protective_factors": [],
    }

    response = ask(
        user_input="How should I manage my fasting glucose of 110?",
        patient_data=patient_ctx,
        conversation_history=[]
    )

    assert response["assistant_response"] == "Mocked guidance response."
    assert not response["safety_flag"]
    assert len(response["retrieved_context"]) > 0
    assert "diabetes" in response["retrieved_context"][0]["id"]
