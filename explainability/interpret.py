import os
import json
import pickle
import pandas as pd
import numpy as np
import warnings

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from phase1.data_loader import load_data, split_features_target
from phase1.preprocessing import preprocess_data
from phase1.models import format_prediction

from phase2.explainability import get_shap_explainer, compute_shap_values, generate_global_explanations, generate_local_explanation
from phase2.counterfactual import simulate_counterfactual, generate_counterfactuals
from phase2.calibration import analyze_calibration
from phase2.fairness_analysis import analyze_fairness

warnings.filterwarnings('ignore')

def main():
    print("Loading Phase 1 artifacts...")
    
    # 1. Load Data
    df = load_data(config.DATA_PATH)
    X, y = split_features_target(df, config.TARGET_COLUMN)
    
    # Needs to match Phase 1 exactly for correct scaling
    X_train_scaled, X_test_scaled, y_train, y_test, _ = preprocess_data(
        X, y, random_state=config.RANDOM_SEED
    )
    
    # 2. Load Model and Scaler
    model_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Run train.py first to generate best_model.pkl and scaler.pkl")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    print(f"Loaded {type(model).__name__}.")
    
    # Optional: XAI output dir
    xai_plot_dir = os.path.join(config.OUTPUT_DIR, "xai_plots")
    os.makedirs(xai_plot_dir, exist_ok=True)
    
    # 3. GLOBAL EXPLAINABILITY
    print("\n--- Running Global Explainability ---")
    explainer = get_shap_explainer(model, X_train_scaled)
    shap_values_test = compute_shap_values(explainer, X_test_scaled)
    
    feature_impacts = generate_global_explanations(
        model, explainer, shap_values_test, X_test_scaled, xai_plot_dir
    )
    print("Top Global Risk Factors:")
    for i, (feat, impact) in enumerate(list(feature_impacts.items())[:5]):
        print(f"  {i+1}. {feat} (Impact: {impact:.4f})")
        
    # 4. CALIBRATION & FAIRNESS
    print("\n--- Running Calibration & Fairness Analysis ---")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_raw = model.decision_function(X_test_scaled)
        y_prob = 1 / (1 + np.exp(-y_pred_raw))
        
    cal_metrics = analyze_calibration(y_test, y_prob, xai_plot_dir)
    print(f"Brier Score: {cal_metrics['brier_score']:.4f}")
    
    # Unscale X_test back to original values for fairness grouping
    X_test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test_scaled), columns=X.columns)
    
    fairness_metrics = analyze_fairness(
        y_true=y_test, 
        y_prob=y_prob, 
        X_raw_test=X_test_unscaled, 
        output_dir=xai_plot_dir,
        sensitive_features=['Age', 'BMI'] # Providing default numeric ones as proxies if others don't exist
    )
    if fairness_metrics:
        print(f"Fairness checks generated for: {', '.join(fairness_metrics.keys())}")
    
    # 5. LOCAL EXPLAINABILITY & COUNTERFACTUALS (Sample Patient)
    print("\n--- Running Local Explainability (Sample Patient) ---")
    
    # Pick a high-risk patient from test set to make it interesting
    high_risk_indices = np.where(y_prob > 0.6)[0]
    if len(high_risk_indices) > 0:
        patient_idx = high_risk_indices[0]
    else:
        patient_idx = 0
        
    X_instance_scaled = X_test_scaled.iloc[patient_idx]
    X_instance_unscaled = X_test_unscaled.iloc[patient_idx]
    
    shap_values_instance = shap_values_test[patient_idx]
    
    local_explanations = generate_local_explanation(
        explainer=explainer, 
        shap_values_instance=shap_values_instance, 
        X_instance=X_instance_unscaled, # Unscaled for readability in plots
        output_dir=xai_plot_dir, 
        patient_id="sample_patient"
    )
    
    prediction_info = format_prediction(y_prob[patient_idx])
    
    # 6. COUNTERFACTUAL ANALYSIS
    print(f"Generating Counterfactuals for Sample Patient...")
    cf_results = generate_counterfactuals(
        model=model, 
        scaler=scaler, 
        X_original_scaled_instance=X_instance_scaled, 
        X_raw_instance=X_instance_unscaled, 
        feature_importance=feature_impacts,
        num_scenarios=2
    )
    
    # 7. FINAL JSON OUTPUT
    final_output = {
        "predicted_risk": round(prediction_info["risk_probability"], 4),
        "risk_level": prediction_info["risk_level"],
        "top_positive_risk_factors": local_explanations["top_positive_risk_factors"],
        "protective_factors": local_explanations["protective_factors"],
        "counterfactual_examples": [cf["explanation"] for cf in cf_results]
    }
    
    print("\n==========================================")
    print("FINAL RISK PROFILING OUPUT")
    print("==========================================")
    print(json.dumps(final_output, indent=2))
    print("==========================================")
    print(f"All plots and graphs saved to: {xai_plot_dir}")

if __name__ == "__main__":
    main()
