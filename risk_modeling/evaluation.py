import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any

def evaluate_model(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculates evaluation metrics given true labels and predicted probabilities."""
    y_pred = (y_prob >= 0.5).astype(int)
    
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob)
    }

def plot_roc_curves(model_probs: Dict[str, np.ndarray], y_true: pd.Series, output_dir: str):
    """Plots ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    for name, y_prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

def plot_feature_importance(model: Any, feature_names: list, model_name: str, output_dir: str):
    """Plots feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20] # Top 20 features
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances - {model_name}")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_importance_{model_name.replace(' ', '_')}.png"))
        plt.close()

def plot_calibration_curves(model_probs: Dict[str, np.ndarray], y_true: pd.Series, output_dir: str):
    """Plots calibration curves."""
    plt.figure(figsize=(10, 8))
    for name, y_prob in model_probs.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "calibration_curves.png"))
    plt.close()

def plot_confusion_matrix(y_true: pd.Series, y_prob: np.ndarray, model_name: str, output_dir: str):
    """Plots confusion matrix heatmap."""
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
    plt.close()
