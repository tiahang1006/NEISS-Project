#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost Multiclass Classifier for NEISS (Body_Part_Category)
Fixes/Enhancements:
  ✓ Split BEFORE encoding (avoid leakage)
  ✓ Sparse One-Hot Encoding (memory-friendly)
  ✓ Label encoding for y (consistent classes)
  ✓ Independent validation set + early stopping (no peeking at test)
  ✓ Class weights for imbalance (better recall on weak classes)
  ✓ Comprehensive metrics (Accuracy, Macro/Weighted F1, OvR AUC)
  ✓ Confusion matrix & classification report saved with proper class names
  ✓ Feature importance saved (Top 20)
  ✓ Model info saved (n_features, n_classes, class_names...)
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# -----------------------------
# Config (EDIT PATHS AS NEEDED)
# -----------------------------
DATA_PATH = "/Users/tianchuhang/Downloads/neisscode/neiss_cleaned_2016_2024.csv"
TARGET_COL = "Body_Part_Category"
CATEGORICAL_COLS = [
    "Sex", "Race", "Age_Category", "Diagnosis_Category",
    "Product_Category", "Season", "Location",
    "Disposition", "Fire_Involvement"
]
OUTDIR = Path("/Users/tianchuhang/Downloads/neisscode/xgboost_output_fixed")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# XGBoost import (install if missing)
# -----------------------------
try:
    from xgboost import XGBClassifier
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    from xgboost import XGBClassifier


# -----------------------------
# Helpers
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    print(f"Loading NEISS cleaned data from: {path}")
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def split_then_encode(
    df: pd.DataFrame,
    cat_cols: list[str],
    target_col: str,
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42
):
    """
    Split data first, then encode to avoid data leakage.
    Create validation set from training data for early stopping.
    """
    X_raw = df[cat_cols].copy()
    y_raw = df[target_col].copy()

    # 1) Train/Test split (stratified)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw
    )
    print(f"Raw train: {X_train_raw.shape}, raw test: {X_test_raw.shape}")

    # 2) Label encode y
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test  = le.transform(y_test_raw)
    class_names = le.classes_
    n_classes = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")

    # 3) From training set, carve out a validation set
    X_train_raw2, X_val_raw, y_train2, y_val = train_test_split(
        X_train_raw, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )
    print(f"Train: {X_train_raw2.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}")

    # 4) Sparse One-Hot Encoding
    ohe = OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=True)
    X_train = ohe.fit_transform(X_train_raw2)
    X_val   = ohe.transform(X_val_raw)
    X_test  = ohe.transform(X_test_raw)
    feature_names = ohe.get_feature_names_out(cat_cols)
    print(f"OHE shapes => Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"n_features after OHE: {len(feature_names):,}")

    return X_train, X_val, X_test, y_train2, y_val, y_test, le, feature_names, class_names


def make_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Generate sample weights based on inverse class frequency to handle class imbalance.
    """
    classes, counts = np.unique(y, return_counts=True)
    # Standard approach: N / (K * n_c)
    weights = {c: (counts.sum() / (len(classes) * counts[i])) for i, c in enumerate(classes)}
    sample_weight = np.array([weights[c] for c in y], dtype=float)
    return sample_weight


def train_xgb(
    X_train, y_train,
    X_val, y_val,
    sample_weight=None,
    random_state: int = 42
):
    """
    Train XGBoost for multiclass classification with early stopping on independent validation set.
    """
    xgb = XGBClassifier(
        n_estimators=200,  # Set large enough max iterations
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=1,  # Show training progress
        early_stopping_rounds=20  # Stop after 20 rounds without improvement
    )

    print("Starting XGBoost training with early stopping...")
    start = time.time()
    xgb.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=True  # Show training progress
    )
    tr_time = time.time() - start
    print(f"XGB training finished in {tr_time:.2f}s")
    if hasattr(xgb, 'best_ntree_limit'):
        print(f"Best iteration: {xgb.best_ntree_limit}")
    return xgb, tr_time


def evaluate_and_save(
    model,
    X_test, y_test,
    class_names: np.ndarray,
    outdir: Path,
    model_name: str,
):
    """
    Unified evaluation and file writing: Accuracy/Macro-F1/Weighted-F1/ROC-AUC(ovr macro/weighted),
    confusion matrix & classification report (with class_names).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    try:
        roc_auc_macro = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        roc_auc_weighted = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc_macro = np.nan
        roc_auc_weighted = np.nan

    # Confusion matrix (fixed order 0..K-1)
    class_order = np.arange(len(class_names))
    cm = confusion_matrix(y_test, y_pred, labels=class_order)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(outdir / f"{model_name}_confusion_matrix.csv", index=True)

    # Classification report (mapped back to readable class names)
    cr_text = classification_report(y_test, y_pred, labels=class_order, target_names=class_names, digits=3)
    (outdir / f"{model_name}_classification_report.txt").write_text(cr_text, encoding="utf-8")

    print("\n=== TEST METRICS ===")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Weighted F1  : {weighted_f1:.4f}")
    print(f"ROC AUCMacro : {roc_auc_macro:.4f}")
    print(f"ROC AUCWeight: {roc_auc_weighted:.4f}")

    results = {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted
    }
    return results, y_pred, y_proba, cm_df


def save_feature_importance(model, feature_names: np.ndarray, outdir: Path, model_name: str):
    """
    Save gain-based feature importance (XGBoost default).
    """
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        fi.head(20).to_csv(outdir / f"{model_name}_top20_features.csv")
        # Also save complete table for further analysis
        fi.to_csv(outdir / f"{model_name}_feature_importance_full.csv")
        print("\nTop-10 features:")
        print(fi.head(10))
        return fi
    else:
        print("No feature_importances_ attribute on model.")
        return None


def save_model_info(
    results: dict,
    feature_names: np.ndarray,
    class_names: np.ndarray,
    training_time: float,
    outdir: Path,
    model_name: str
):
    info = {
        "model_type": "XGBoost",
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "roc_auc_macro": results["roc_auc_macro"],
        "roc_auc_weighted": results["roc_auc_weighted"],
        "training_time_sec": training_time,
        "n_features": len(feature_names),
        "n_classes": len(class_names),
        "class_names": "|".join(class_names.tolist())
    }
    pd.DataFrame([info]).to_csv(outdir / f"{model_name}_model_info.csv", index=False)


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 90)
    print("XGBOOST MULTICLASS (NEISS Body_Part_Category) — FIXED/ENHANCED")
    print("Fixes: split-before-encode, sparse OHE, label-encoded y,")
    print("       independent val (early stopping), class weights, robust eval+saving")
    print("=" * 90)

    df = load_data(DATA_PATH)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     label_encoder, feature_names, class_names) = split_then_encode(
        df, CATEGORICAL_COLS, TARGET_COL, test_size=0.20, val_size=0.10, random_state=42
    )

    # Class weights (to handle imbalance/low recall)
    sample_weight = make_class_weights(y_train)

    # Training (early stopping on independent validation set)
    model, tr_time = train_xgb(
        X_train, y_train,
        X_val, y_val,
        sample_weight=sample_weight,
        random_state=42
    )

    # Test evaluation & save
    results, y_pred, y_proba, cm_df = evaluate_and_save(
        model, X_test, y_test, class_names, OUTDIR, model_name="xgboost"
    )
    fi = save_feature_importance(model, feature_names, OUTDIR, model_name="xgboost")
    save_model_info(results, feature_names, class_names, tr_time, OUTDIR, model_name="xgboost")

    print("\nArtifacts saved to:", OUTDIR.resolve())
    print("- xgboost_confusion_matrix.csv")
    print("- xgboost_classification_report.txt")
    print("- xgboost_top20_features.csv & xgboost_feature_importance_full.csv")
    print("- xgboost_model_info.csv")
    print("\n✅ Done.")

    return model, results, (label_encoder, feature_names, class_names), (y_pred, y_proba, cm_df)


if __name__ == "__main__":
    model, results, meta, eval_artifacts = main()
