import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "/Users/tianchuhang/Downloads/neisscode/neiss_cleaned_2016_2024.csv"
TARGET_COL = "Body_Part_Category"
CATEGORICAL_COLS = [
    "Sex", "Race", "Age_Category", "Diagnosis_Category",
    "Product_Category", "Season", "Location",
    "Disposition", "Fire_Involvement"
]
OUTDIR = Path("/Users/tianchuhang/Downloads/neisscode/rf_output")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper Functions
# -----------------------------
def load_and_prepare_data():
    """Load the cleaned data and prepare for modeling"""
    print("Loading cleaned NEISS data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {df.shape}")
    return df

def encode_categorical_variables(df):
    """Encode categorical variables using one-hot encoding"""
    print("Encoding categorical variables...")
    
    # Create dummy variables for categorical columns
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    
    print(f"Encoded data shape: {df_encoded.shape}")
    return df_encoded

def prepare_model_data(df_encoded):
    """Prepare features and target for modeling"""
    print("Preparing model data...")
    
    # Separate features and target
    X = df_encoded.drop(TARGET_COL, axis=1)
    y = df_encoded[TARGET_COL]
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    print(f"Features: {X.shape[1]}")
    print(f"Target classes: {y.nunique()}")
    print(f"Class distribution:")
    print(y.value_counts())
    
    return X, y, feature_names

def evaluate_model_comprehensive(y_true, y_pred, y_pred_proba, model_name, class_order):
    """Comprehensive model evaluation"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # ROC AUC
    try:
        roc_auc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        roc_auc_macro = roc_auc_weighted = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
    cm_df.to_csv(OUTDIR / f"{model_name}_confusion_matrix.csv")
    
    # Classification report
    with open(OUTDIR / f"{model_name}_classification_report.txt", 'w') as f:
        f.write(classification_report(y_true, y_pred, labels=class_order))
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'roc_auc_macro': roc_auc_macro,
        'roc_auc_weighted': roc_auc_weighted,
        'confusion_matrix': cm_df
    }

def train_and_evaluate_rf(model, model_name, X_train, X_test, y_train, y_test, class_order):
    """Train Random Forest model and return comprehensive evaluation"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    training_time = time.time() - start_time
    
    # Evaluate
    results = evaluate_model_comprehensive(y_test, y_pred, y_pred_proba, model_name, class_order)
    results['training_time'] = training_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    
    return model, results, y_pred, y_pred_proba

def analyze_feature_importance(model, model_name, feature_names):
    """Analyze and save feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save full importance
        importance_df.to_csv(OUTDIR / f"{model_name}_feature_importance_full.csv", index=False)
        
        # Save top 20 features
        top20 = importance_df.head(20)
        top20.to_csv(OUTDIR / f"{model_name}_top20_features.csv", index=False)
        
        print(f"\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(top20.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        return importance_df
    else:
        print(f"{model_name} does not have feature_importances_ attribute")
        return None

def save_model_info(model, model_name, results, training_time):
    """Save model information"""
    model_info = {
        'model_name': model_name,
        'n_estimators': getattr(model, 'n_estimators', 'N/A'),
        'max_depth': getattr(model, 'max_depth', 'N/A'),
        'min_samples_split': getattr(model, 'min_samples_split', 'N/A'),
        'min_samples_leaf': getattr(model, 'min_samples_leaf', 'N/A'),
        'random_state': getattr(model, 'random_state', 'N/A'),
        'accuracy': results['accuracy'],
        'macro_f1': results['macro_f1'],
        'weighted_f1': results['weighted_f1'],
        'training_time': training_time
    }
    
    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_csv(OUTDIR / f"{model_name}_model_info.csv", index=False)
    
    print(f"\nModel information saved to {OUTDIR / f'{model_name}_model_info.csv'}")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("="*80)
    print("RANDOM FOREST MODEL TRAINING")
    print("="*80)
    
    # Load and prepare data
    df = load_and_prepare_data()
    df_encoded = encode_categorical_variables(df)
    X, y, feature_names = prepare_model_data(df_encoded)
    
    # Train-test split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Get class order for consistent evaluation
    class_order = sorted(y.unique())
    print(f"Class order: {class_order}")
    
    # Random Forest Model
    print("\n" + "="*60)
    print("RANDOM FOREST CLASSIFIER")
    print("="*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train and evaluate
    rf_trained, rf_results, rf_pred, rf_pred_proba = train_and_evaluate_rf(
        rf_model, "random_forest", X_train, X_test, y_train, y_test, class_order
    )
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    rf_importance = analyze_feature_importance(rf_trained, "random_forest", feature_names)
    
    # Save model info
    save_model_info(rf_trained, "random_forest", rf_results, rf_results['training_time'])
    
    print("\n" + "="*80)
    print("RANDOM FOREST TRAINING COMPLETED")
    print("="*80)
    print(f"Results saved to: {OUTDIR}")
    print(f"Accuracy: {rf_results['accuracy']:.4f}")
    print(f"Macro F1: {rf_results['macro_f1']:.4f}")
    print(f"Weighted F1: {rf_results['weighted_f1']:.4f}")
    if rf_results['roc_auc_macro'] is not None:
        print(f"ROC AUC (Macro): {rf_results['roc_auc_macro']:.4f}")
        print(f"ROC AUC (Weighted): {rf_results['roc_auc_weighted']:.4f}")

if __name__ == "__main__":
    main()
