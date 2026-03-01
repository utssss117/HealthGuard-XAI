import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List

def get_shap_explainer(model: Any, X_train: pd.DataFrame):
    """
    Returns an appropriate SHAP explainer based on the model type.
    """
    model_type = type(model).__name__
    if 'RandomForest' in model_type or 'XGB' in model_type:
        explainer = shap.TreeExplainer(model)
    elif 'Logistic' in model_type or 'Linear' in model_type:
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # Fallback to KernelExplainer for models like MLP or StackingEnsemble
        # We sample the background dataset to speed up KernelExplainer
        background = shap.sample(X_train, min(50, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
    return explainer

def compute_shap_values(explainer, X: pd.DataFrame):
    shap_values = explainer.shap_values(X)
    # SHAP values might be a list (one array per class) for some models/explainers.
    # We want the values for the positive class (class 1).
    if isinstance(shap_values, list) and len(shap_values) > 1:
        return shap_values[1]
    # some TreeExplainer return an array of shape (n_samples, n_features, n_classes)
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        return shap_values[:, :, 1]
    return shap_values

def generate_global_explanations(model: Any, explainer: Any, shap_values_test: np.ndarray, X_test: pd.DataFrame, output_dir: str) -> Dict[str, float]:
    """
    Computes global SHAP explanations, generates plots, and returns feature impacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. SHAP Summary Plot
    plt.figure()
    shap.summary_plot(shap_values_test, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Global Feature Importance (Bar Plot)
    plt.figure()
    shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Calculate Feature Impact Ranking
    mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
    feature_impacts = {feature: float(impact) for feature, impact in zip(X_test.columns, mean_abs_shap)}
    # Sort by impact
    feature_impacts = dict(sorted(feature_impacts.items(), key=lambda item: item[1], reverse=True))
    
    return feature_impacts

def generate_local_explanation(explainer: Any, shap_values_instance: np.ndarray, X_instance: pd.Series, output_dir: str, patient_id: str = "sample") -> Dict[str, List[str]]:
    """
    Generates local SHAP explanations for a single instance.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Base value
    if hasattr(explainer, 'expected_value'):
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
            expected_value = expected_value[1]
    else:
        # Fallback if explainer doesn't expose expected_value easily
        expected_value = 0.5 

    # Generate Force Plot (save as HTML or image if possible, but force_plot is interactive HTML by default. 
    # We can use waterfall plot for static images in newer shap versions, or save force plot as html)
    
    # We will create a waterfall plot which is static and good for saving as PNG
    plt.figure(figsize=(10, 6))
    
    # To use waterfall, we need to create an Explanation object
    explanation = shap.Explanation(values=shap_values_instance, 
                                   base_values=expected_value, 
                                   data=X_instance.values, 
                                   feature_names=X_instance.index.tolist())
    
    try:
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_waterfall_{patient_id}.png"), dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Warning: Could not save waterfall plot: {e}")
    plt.close()

    # Determine Top Positive and Negative Contributors
    # Feature values and their SHAP contributions
    contributions = [(feature, val, shap_val) for feature, val, shap_val in 
                     zip(X_instance.index, X_instance.values, shap_values_instance)]
    
    # Sort by SHAP value
    contributions.sort(key=lambda x: x[2])
    
    # Protective (Negative SHAP values)
    protective_factors = [f"{f} ({v:.2f})" for f, v, s in contributions if s < 0]
    protective_factors.reverse() # Most protective first
    
    # Risk increasing (Positive SHAP values)
    positive_risk_factors = [f"{f} ({v:.2f})" for f, v, s in contributions if s > 0]
    positive_risk_factors.reverse() # Highest risk first
    
    return {
        "top_positive_risk_factors": positive_risk_factors[:5], # Top 5
        "protective_factors": protective_factors[:5]           # Top 5
    }
