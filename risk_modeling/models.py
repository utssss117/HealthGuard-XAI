from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from typing import Dict, Any

def get_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns a dictionary of uninitialized models."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_estimators=100),
        "XGBoost": XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
        "MLP": MLPClassifier(random_state=random_state, max_iter=1000, early_stopping=True)
    }
    
    # Stacking Ensemble
    estimators = [
        ('rf', models["Random Forest"]),
        ('xgb', models["XGBoost"]),
        ('mlp', models["MLP"])
    ]
    models["Stacking Ensemble"] = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )
    
    return models

def format_prediction(probability: float) -> Dict[str, Any]:
    """Formats raw probability into risk score and category."""
    if probability <= 0.33:
        risk_level = "Low"
    elif probability <= 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"
        
    return {
        "risk_probability": float(probability),
        "risk_level": risk_level
    }
