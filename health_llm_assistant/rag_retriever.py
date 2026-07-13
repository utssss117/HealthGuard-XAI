"""
rag_retriever.py
────────────────
Lightweight, offline Retrieval-Augmented Generation (RAG) retriever for clinical guidelines.
Uses scikit-learn's TF-IDF vectorizer to find relevant guideline chunks matching patient queries.
"""

import os
import json
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ClinicalRAGRetriever:
    """
    Retrieves official medical guidelines using TF-IDF and Cosine Similarity.
    """
    def __init__(self, guidelines_path: str = None):
        if guidelines_path is None:
            guidelines_path = os.path.join(os.path.dirname(__file__), "clinical_guidelines.json")
        
        self.guidelines_path = guidelines_path
        self.guidelines: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        self._load_and_index_guidelines()

    def _load_and_index_guidelines(self) -> None:
        """Loads guidelines from JSON and fits the TF-IDF matrix."""
        if not os.path.exists(self.guidelines_path):
            raise FileNotFoundError(f"Guidelines file not found at: {self.guidelines_path}")
            
        with open(self.guidelines_path, "r", encoding="utf-8") as f:
            self.guidelines = json.load(f)
            
        if not self.guidelines:
            raise ValueError("Clinical guidelines database is empty.")
            
        # Fit vectorizer on guideline contents
        corpus = [item["content"] for item in self.guidelines]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 2, threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Retrieve top_k guidelines that match the query above a minimum similarity threshold.
        """
        if not query.strip() or self.tfidf_matrix is None:
            return []

        # Vectorize user query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Sort indices by descending score
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append({
                    "id": self.guidelines[idx]["id"],
                    "title": self.guidelines[idx]["title"],
                    "content": self.guidelines[idx]["content"],
                    "score": score
                })
            if len(results) >= top_k:
                break
                
        return results

    def format_context_for_prompt(self, matched_docs: List[Dict[str, Any]]) -> str:
        """Formats matched documents into a string context block for the system prompt."""
        if not matched_docs:
            return "No specific official guidelines matches found for this query."
            
        context_parts = []
        for i, doc in enumerate(matched_docs, 1):
            context_parts.append(f"Source [{i}]: {doc['title']}\nGuideline: {doc['content']}")
            
        return "\n\n".join(context_parts)
