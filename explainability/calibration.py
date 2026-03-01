import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import numpy as np
from typing import Dict, Any

def analyze_calibration(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str) -> Dict[str, float]:
    """
    Computes Brier score and generates calibration curve.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve (Reliability Diagram)\nBrier Score: {brier:.4f}')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "brier_score": brier
    }
