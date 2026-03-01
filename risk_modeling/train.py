import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from phase1.data_loader import load_data, split_features_target
from phase1.preprocessing import preprocess_data
from phase1.models import get_models
from phase1.evaluation import (
    evaluate_model, plot_roc_curves, plot_feature_importance, 
    plot_calibration_curves, plot_confusion_matrix
)

def main():
    print("Loading data...")
    df = load_data(config.DATA_PATH)
    X, y = split_features_target(df, config.TARGET_COLUMN)
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        X, y, random_state=config.RANDOM_SEED
    )
    
    models = get_models(config.RANDOM_SEED)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    
    results = []
    test_probs = {}
    best_model_name = None
    best_auc = 0
    best_model = None

    print(f"Training and Evaluating {len(models)} models using {config.CV_FOLDS}-fold CV...")
    
    for name, model in models.items():
        print(f"--- {name} ---")
        
        # Cross-validation
        cv_aucs = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model_clone = get_models(config.RANDOM_SEED)[name]
            model_clone.fit(X_cv_train, y_cv_train)
            
            if hasattr(model_clone, "predict_proba"):
                preds = model_clone.predict_proba(X_cv_val)[:, 1]
            else:
                preds = model_clone.decision_function(X_cv_val)
                preds = 1 / (1 + np.exp(-preds)) # sigmoid
            
            cv_aucs.append(roc_auc_score(y_cv_val, preds))
            
        print(f"  CV AUC:     {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs):.4f})")
        
        # Train on full train set
        model.fit(X_train, y_train)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_raw = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_pred_raw))
            
        test_probs[name] = y_prob
        
        # Evaluate on test set
        metrics = evaluate_model(y_test, y_prob)
        metrics["Model"] = name
        metrics["CV_AUC_mean"] = np.mean(cv_aucs)
        metrics["CV_AUC_std"] = np.std(cv_aucs)
        results.append(metrics)
        
        print(f"  Test AUC:   {metrics['ROC-AUC']:.4f}")
        
        # Plots
        plot_confusion_matrix(y_test, y_prob, name, config.PLOTS_DIR)
        
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, list(X.columns), name, config.PLOTS_DIR)
        
        # Model selection (based on test AUC for now, could be CV AUC)
        if metrics["ROC-AUC"] > best_auc:
            best_auc = metrics["ROC-AUC"]
            best_model_name = name
            best_model = model

    print("\nPlotting combined evaluation curves...")
    plot_roc_curves(test_probs, y_test, config.PLOTS_DIR)
    plot_calibration_curves(test_probs, y_test, config.PLOTS_DIR)
    
    # Save results
    results_df = pd.DataFrame(results)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Brier Score", "CV_AUC_mean", "CV_AUC_std"]
    results_df = results_df[cols]
    
    results_path = os.path.join(config.METRICS_DIR, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    
    print("\n--- Final Results ---")
    print(results_df.to_string(index=False))
    
    print(f"\nBest Model: {best_model_name} (AUC: {best_auc:.4f})")
    
    # Save best model
    model_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {model_path}")
    
    # Save scaler for future predictions
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
if __name__ == "__main__":
    main()
