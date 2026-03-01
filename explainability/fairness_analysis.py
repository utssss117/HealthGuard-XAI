import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

def analyze_fairness(y_true: pd.Series, y_prob: np.ndarray, X_raw_test: pd.DataFrame, output_dir: str, sensitive_features: List[str] = None) -> Dict[str, Any]:
    """
    Analyzes prediction probability distribution across demographic groups.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    df_eval = X_raw_test.copy()
    df_eval['Actual'] = y_true.values
    df_eval['Predicted_Prob'] = y_prob
    
    if not sensitive_features:
        # Try to infer sensitive features based on common names if none provided
        candidates = ['Age', 'age', 'Gender', 'gender', 'Sex', 'sex', 'Race', 'race', 'Ethnicity', 'ethnicity']
        sensitive_features = [col for col in candidates if col in df_eval.columns]
    
    for feature in sensitive_features:
        if feature not in df_eval.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # If numeric and many unique values (like Age), bin it
        if pd.api.types.is_numeric_dtype(df_eval[feature]) and df_eval[feature].nunique() > 10:
            df_eval[f'{feature}_Group'] = pd.qcut(df_eval[feature], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            plot_feature = f'{feature}_Group'
        else:
            plot_feature = feature
            
        sns.violinplot(x=plot_feature, y='Predicted_Prob', data=df_eval, inner='quartile', palette="Set2")
        plt.title(f'Prediction Probability Distribution by {feature}')
        plt.ylabel('Predicted Risk Probability')
        
        # Find means for text output
        group_means = df_eval.groupby(plot_feature, observed=False)['Predicted_Prob'].mean().to_dict()
        results[feature] = {
            "group_mean_probabilities": {str(k): float(v) for k, v in group_means.items()}
        }
        
        plt.tight_layout()
        filename = f"fairness_distribution_{feature}.png".replace(" ", "_")
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
    return results
