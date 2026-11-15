"""Train machine learning baseline classifiers and the Proposed Hybrid AI ensemble.
Computes and records rigorous experimental metrics:
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrices
- Full per-class classification reports
- Feature importances
Saves trained artifacts in models/ and evaluation logs in reports/.
"""
from pathlib import Path
import json
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "synthetic_model_data.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def get_models():
    """
    Construct the models evaluated in the research paper:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
    4. Proposed Hybrid AI Model (Soft-Voting Ensemble combining LR, RF, and XGBoost)
    """
    # 1. Logistic Regression Pipeline with Feature Scaling
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])

    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=9,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )

    # 3. XGBoost Classifier
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42
        )
        xgb_name = "XGBoost"
    except Exception as e:
        print(f"Warning: xgboost import error ({e}), using HistGradientBoostingClassifier as fallback.")
        from sklearn.ensemble import HistGradientBoostingClassifier
        xgb_model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.07,
            max_leaf_nodes=20,
            random_state=42
        )
        xgb_name = "HistGradientBoosting (XGBoost Fallback)"

    # 4. Proposed Hybrid AI Model (Multi-Model Ensemble)
    # Combines linear decision boundaries (LR) with tree partitioning (RF) and gradient boosted residuals (XGB)
    hybrid_model = VotingClassifier(
        estimators=[
            ("lr", lr_pipeline),
            ("rf", rf_model),
            ("boosted", xgb_model)
        ],
        voting="soft",
        weights=[1.0, 1.5, 2.0]
    )

    return {
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_model,
        xgb_name: xgb_model,
        "Proposed Hybrid AI Model": hybrid_model
    }


def train_and_evaluate(random_seed: int = 42):
    """Execute complete training and evaluation protocol."""
    if not DATA_PATH.exists():
        from src.generate_demo_data import generate_all_data
        generate_all_data(random_seed=random_seed)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    target_col = "purchase_intent"
    if target_col not in df.columns:
        # Fallback for earlier proxy column if present
        target_col = "purchase_intent_proxy" if "purchase_intent_proxy" in df.columns else df.columns[-1]

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )

    models_dict = get_models()

    metrics_list = []
    confusion_dict = {}
    reports_dict = {}
    feature_importance_dict = {}

    print(f"\n=================================================================")
    print(f"TRAINING & EVALUATION PIPELINE (Random Seed: {random_seed})")
    print(f"Dataset Size: {len(df)} samples | Features: {len(feature_cols)} | Test Split: {len(y_test)} samples")
    print(f"=================================================================\n")

    for model_name, model in models_dict.items():
        print(f"--> Training {model_name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc_val = round(roc_auc_score(y_test, y_proba) * 100, 2)
        else:
            auc_val = None

        # Metrics
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        prec = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
        rec = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)

        metrics_list.append({
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc_val
        })

        # Confusion Matrix [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_test, y_pred)
        confusion_dict[model_name] = {
            "matrix": cm.tolist(),
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1])
        }

        # Detailed Classification Report
        cls_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reports_dict[model_name] = cls_rep

        # Feature Importance (if model provides it)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance_dict[model_name] = {
                feat: round(float(imp), 4) for feat, imp in zip(feature_cols, importances)
            }

        # Save model artifact
        safe_key = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        joblib.dump(model, MODELS_DIR / f"{safe_key}.joblib")

        print(f"    Accuracy: {acc}% | Precision: {prec}% | Recall: {rec}% | F1-Score: {f1}% | AUC: {auc_val}%")

    # Record feature importances for Random Forest and XGBoost if available
    for key, model in models_dict.items():
        if hasattr(model, "named_steps"):
            pass
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance_dict[key] = {
                feat: round(float(imp), 4) for feat, imp in zip(feature_cols, importances)
            }

    # Consolidated Results Object
    evaluation_payload = {
        "dataset_name": "synthetic_model_data.csv",
        "dataset_type": "Synthetic Demonstration Data (clearly labelled)",
        "total_samples": len(df),
        "test_samples": len(y_test),
        "target_variable": target_col,
        "split_ratio": "80% Train / 20% Test (Stratified)",
        "random_seed": random_seed,
        "features": feature_cols,
        "metrics": metrics_list
    }

    # Save reports
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)

    with open(REPORTS_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(confusion_dict, f, indent=2)

    with open(REPORTS_DIR / "classification_reports.json", "w", encoding="utf-8") as f:
        json.dump(reports_dict, f, indent=2)

    with open(REPORTS_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(feature_importance_dict, f, indent=2)

    print(f"\nSaved evaluation reports and trained models to:")
    print(f"  - {REPORTS_DIR / 'model_metrics.json'}")
    print(f"  - {REPORTS_DIR / 'confusion_matrices.json'}")
    print(f"  - {MODELS_DIR}/*.joblib")
    return evaluation_payload


if __name__ == "__main__":
    train_and_evaluate()
