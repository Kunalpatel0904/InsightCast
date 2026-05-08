"""churn_model.py — XGBoost churn classifier + SHAP explainability"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings("ignore")


def train_churn_model(X, y, feature_names):
    """Train XGBoost, return model + metrics dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        early_stopping_rounds=15,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "auc_roc":   round(roc_auc_score(y_test, y_proba) * 100, 2),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Cross-val AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                      use_label_encoder=False, eval_metric="logloss",
                      random_state=42, verbosity=0),
        X, y, cv=cv, scoring="roc_auc")
    metrics["cv_auc_mean"] = round(cv_scores.mean() * 100, 2)
    metrics["cv_auc_std"]  = round(cv_scores.std() * 100, 2)

    return model, metrics


def get_shap_values(model, X, feature_names):
    """Return explainer + shap_values array."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def predict_churn(model, X, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    flags = (proba >= threshold).astype(int)
    return proba, flags
