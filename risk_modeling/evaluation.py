"""
evaluation.py
─────────────
Model evaluation utilities for HealthGuard-XAI.

Provides both:
  - matplotlib static plots (for training/saving)
  - Plotly interactive charts (for the Streamlit dashboard)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss, roc_curve,
)
from sklearn.calibration import calibration_curve
from typing import Dict, Any, List


# ── Core Metrics ──────────────────────────────────────────────────────────────

def evaluate_model(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate a full suite of binary classification metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Precision":   round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":      round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-score":    round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":     round(roc_auc_score(y_true, y_prob), 4),
        "Brier Score": round(brier_score_loss(y_true, y_prob), 4),
    }


# ── Matplotlib Static Plots (used during training) ────────────────────────────

def plot_roc_curves(
    model_probs: Dict[str, np.ndarray],
    y_true: pd.Series,
    output_dir: str,
) -> None:
    """Save combined ROC curves as PNG."""
    plt.figure(figsize=(10, 8))
    for name, y_prob in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    model_name: str,
    output_dir: str,
) -> None:
    """Save feature importance bar chart as PNG (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances — {model_name}")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"feature_importance_{model_name.replace(' ', '_')}.png"),
        dpi=150,
    )
    plt.close()


def plot_calibration_curves(
    model_probs: Dict[str, np.ndarray],
    y_true: pd.Series,
    output_dir: str,
) -> None:
    """Save calibration curves as PNG."""
    plt.figure(figsize=(10, 8))
    for name, y_prob in model_probs.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves — All Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: str,
) -> None:
    """Save confusion matrix heatmap as PNG."""
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png"),
        dpi=150,
    )
    plt.close()


# ── Plotly Interactive Charts (used in Streamlit dashboard) ───────────────────

_COLOR_MAP = {
    "Logistic Regression": "#4CC9F0",
    "Random Forest":        "#4361EE",
    "XGBoost":              "#7209B7",
    "MLP":                  "#F72585",
    "Stacking Ensemble":    "#3A0CA3",
}

_COLORS = list(_COLOR_MAP.values())


def get_plotly_roc_curves(
    model_probs: Dict[str, np.ndarray],
    y_true: pd.Series,
) -> go.Figure:
    """Return an interactive Plotly ROC curve figure."""
    fig = go.Figure()

    for i, (name, y_prob) in enumerate(model_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = _COLOR_MAP.get(name, _COLORS[i % len(_COLORS)])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=color, width=2.5),
            hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1),
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text="ROC Curves — All Models", font=dict(size=16)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        height=420,
    )
    return fig


def get_plotly_confusion_matrix(
    y_true: pd.Series,
    y_prob: np.ndarray,
    model_name: str = "Best Model",
) -> go.Figure:
    """Return an interactive Plotly confusion matrix heatmap."""
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Diabetes", "Diabetes"]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, "#1a1a2e"], [1, "#4361EE"]],
        showscale=False,
        text=cm.astype(str),
        texttemplate="%{text}",
        textfont=dict(size=22, color="white"),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=16)),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def get_plotly_calibration_curves(
    model_probs: Dict[str, np.ndarray],
    y_true: pd.Series,
) -> go.Figure:
    """Return an interactive Plotly calibration curve figure."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfect Calibration",
        line=dict(color="rgba(255,255,255,0.4)", dash="dash", width=1.5),
        hoverinfo="skip",
    ))

    for i, (name, y_prob) in enumerate(model_probs.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        color = _COLOR_MAP.get(name, _COLORS[i % len(_COLORS)])
        fig.add_trace(go.Scatter(
            x=prob_pred, y=prob_true, mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=7),
            hovertemplate=f"<b>{name}</b><br>Predicted: %{{x:.2f}}<br>Actual: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Calibration Curves — All Models", font=dict(size=16)),
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        height=420,
    )
    return fig


def get_plotly_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Return a grouped bar chart comparing all models across metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    fig = go.Figure()

    for i, row in results_df.iterrows():
        color = _COLOR_MAP.get(row["Model"], _COLORS[i % len(_COLORS)])
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=color,
            text=[f"{row[m]:.3f}" for m in metrics],
            textposition="outside",
            hovertemplate=f"<b>{row['Model']}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="Model Performance Comparison", font=dict(size=16)),
        yaxis=dict(range=[0, 1.1], title="Score"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        height=450,
    )
    return fig
