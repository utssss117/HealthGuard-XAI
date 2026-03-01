import pandas as pd
import numpy as np
from typing import Dict, Any, List

def simulate_counterfactual(model: Any, scaler: Any, X_original_scaled_instance: pd.Series, X_raw_instance: pd.Series, modifications: Dict[str, float]) -> Dict[str, Any]:
    """
    Simulates a counterfactual scenario by modifying specific features and calculating the change in predicted risk.
    """
    # 1. Base prediction
    if hasattr(model, "predict_proba"):
        base_prob = model.predict_proba(X_original_scaled_instance.to_frame().T)[0, 1]
    else:
        base_pred_raw = model.decision_function(X_original_scaled_instance.to_frame().T)[0]
        base_prob = 1 / (1 + np.exp(-base_pred_raw))
        
    # 2. Apply modifications to the RAW input
    X_counterfactual_raw = X_raw_instance.copy()
    for feature, new_value in modifications.items():
        if feature in X_counterfactual_raw.index:
            X_counterfactual_raw[feature] = new_value
        else:
            print(f"Warning: Feature '{feature}' not found in instance. Skipping modification.")
            
    # 3. Scale the new counterfactual input
    X_counterfactual_scaled = pd.DataFrame(
        scaler.transform(X_counterfactual_raw.to_frame().T),
        columns=X_counterfactual_raw.index
    )
    
    # 4. New prediction
    if hasattr(model, "predict_proba"):
        new_prob = model.predict_proba(X_counterfactual_scaled)[0, 1]
    else:
        new_pred_raw = model.decision_function(X_counterfactual_scaled)[0]
        new_prob = 1 / (1 + np.exp(-new_pred_raw))
        
    # 5. Calculate difference
    risk_diff = new_prob - base_prob
    risk_diff_percent = risk_diff * 100
    
    # 6. Format explanation
    mod_strings = []
    for f, v in modifications.items():
        if f in X_raw_instance.index:
            old_v = X_raw_instance[f]
            mod_strings.append(f"{f} changed from {old_v:.1f} → {v:.1f}")
            
    mod_str = " and ".join(mod_strings)
    
    if risk_diff < 0:
        direction = "decreases"
        amount = abs(risk_diff_percent)
    else:
        direction = "increases"
        amount = risk_diff_percent
        
    explanation = f"If {mod_str}, predicted risk {direction} by {amount:.1f}%."
    
    return {
        "modifications": modifications,
        "base_risk": base_prob,
        "new_risk": new_prob,
        "risk_difference": risk_diff,
        "risk_difference_percentage": risk_diff_percent,
        "explanation": explanation
    }
    
def generate_counterfactuals(model: Any, scaler: Any, X_original_scaled_instance: pd.Series, X_raw_instance: pd.Series, feature_importance: Dict[str, float], num_scenarios: int = 2) -> List[Dict[str, Any]]:
    """
    Automatically generates a few counterfactual scenarios based on top important features.
    (e.g., reducing a top risk factor by 20%)
    """
    counterfactuals = []
    
    # Get top 3 features by global impact to try modifications on
    top_features = list(feature_importance.keys())[:3]
    
    for feature in top_features:
        if len(counterfactuals) >= num_scenarios:
            break
            
        current_val = X_raw_instance[feature]
        # Skip if value is already 0 or very small
        if current_val <= 0.1:
            continue
            
        # Strategy: Reduce the feature by 20%
        new_val = current_val * 0.8
        
        # Determine if it's a binary or discrete feature by heuristics (for simplicity, just round to 1 decimal)
        if hasattr(current_val, "is_integer") and current_val.is_integer() or current_val % 1 == 0:
             new_val = round(new_val)
        else:
             new_val = round(new_val, 1)
             
        # Skip if no meaningful change
        if new_val == current_val:
            continue
            
        mods = {feature: new_val}
        res = simulate_counterfactual(model, scaler, X_original_scaled_instance, X_raw_instance, mods)
        counterfactuals.append(res)
        
    return counterfactuals
