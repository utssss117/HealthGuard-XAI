"""
train.py
────────
HealthGuard-XAI — ML Training Pipeline

Trains and evaluates 5 models on the PIMA Diabetes dataset:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - MLP Neural Network
  - Stacking Ensemble

Best model (by ROC-AUC) is saved to outputs/models/best_model.pkl
Scaler is saved to outputs/models/scaler.pkl

Run from project root:
    python -m risk_modeling.train
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from risk_modeling.data_loader import load_data, split_features_target
from risk_modeling.preprocessing import preprocess_data
from risk_modeling.models import get_models
from risk_modeling.evaluation import (
    evaluate_model,
    plot_roc_curves,
    plot_feature_importance,
    plot_calibration_curves,
    plot_confusion_matrix,
)


def main():
    print("=" * 60)
    print("  HealthGuard-XAI — Model Training Pipeline")
    print("=" * 60)

    # ── Load Data ────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_data(config.DATA_PATH)
    X, y = split_features_target(df, config.TARGET_COLUMN)
    print(f"      Dataset shape: {df.shape} | Positive rate: {y.mean():.2%}")

    # ── Preprocess ───────────────────────────────────────────────
    print("\n[2/5] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, random_state=config.RANDOM_SEED
    )
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

    # ── Cross-Validation + Training ──────────────────────────────
    models = get_models(config.RANDOM_SEED)
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )

    results = []
    test_probs = {}
    best_model_name = None
    best_auc = 0.0
    best_model = None

    print(f"\n[3/5] Training {len(models)} models with {config.CV_FOLDS}-fold CV...\n")

    for name, model in models.items():
        print(f"  > {name}")

        # Cross-validation
        cv_aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train = X_train.iloc[train_idx]
            X_cv_val   = X_train.iloc[val_idx]
            y_cv_train = y_train.iloc[train_idx]
            y_cv_val   = y_train.iloc[val_idx]

            clone = get_models(config.RANDOM_SEED)[name]
            clone.fit(X_cv_train, y_cv_train)

            if hasattr(clone, "predict_proba"):
                preds = clone.predict_proba(X_cv_val)[:, 1]
            else:
                raw = clone.decision_function(X_cv_val)
                preds = 1 / (1 + np.exp(-raw))

            cv_aucs.append(roc_auc_score(y_cv_val, preds))

        print(f"    CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

        # Train on full training set
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            raw = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-raw))

        test_probs[name] = y_prob
        metrics = evaluate_model(y_test, y_prob)
        metrics["Model"]       = name
        metrics["CV_AUC_mean"] = float(np.mean(cv_aucs))
        metrics["CV_AUC_std"]  = float(np.std(cv_aucs))
        results.append(metrics)

        print(f"    Test AUC: {metrics['ROC-AUC']:.4f}")

        # Per-model plots
        plot_confusion_matrix(y_test, y_prob, name, config.PLOTS_DIR)
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(model, list(X.columns), name, config.PLOTS_DIR)

        # Track best
        if metrics["ROC-AUC"] > best_auc:
            best_auc        = metrics["ROC-AUC"]
            best_model_name = name
            best_model      = model

    # ── Combined Plots ───────────────────────────────────────────
    print("\n[4/5] Generating evaluation plots...")
    plot_roc_curves(test_probs, y_test, config.PLOTS_DIR)
    plot_calibration_curves(test_probs, y_test, config.PLOTS_DIR)

    # ── Save Results ─────────────────────────────────────────────
    print("\n[5/5] Saving model, scaler, and metrics...")

    results_df = pd.DataFrame(results)
    cols = [
        "Model", "Accuracy", "Precision", "Recall",
        "F1-score", "ROC-AUC", "Brier Score", "CV_AUC_mean", "CV_AUC_std",
    ]
    results_df = results_df[cols]
    results_path = os.path.join(config.METRICS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    model_path  = os.path.join(config.MODEL_DIR, "best_model.pkl")
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\n" + "=" * 60)
    print(f"  Best Model : {best_model_name}  (AUC: {best_auc:.4f})")
    print(f"  Saved to   : {model_path}")
    print("=" * 60)
    print("\n[DONE] Training complete!\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
