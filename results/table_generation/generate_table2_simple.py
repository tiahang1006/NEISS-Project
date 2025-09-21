#!/usr/bin/env python3
"""
Generate Table 2: Performance metrics comparison across models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

def extract_metrics_from_classification_report(file_path):
    """Extract metrics from classification report text file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract accuracy
    accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', content)
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None
    
    # Extract macro avg
    macro_match = re.search(r'macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', content)
    macro_precision = float(macro_match.group(1)) if macro_match else None
    macro_recall = float(macro_match.group(2)) if macro_match else None
    macro_f1 = float(macro_match.group(3)) if macro_match else None
    
    # Extract weighted avg
    weighted_match = re.search(r'weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', content)
    weighted_precision = float(weighted_match.group(1)) if weighted_match else None
    weighted_recall = float(weighted_match.group(2)) if weighted_match else None
    weighted_f1 = float(weighted_match.group(3)) if weighted_match else None
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def calculate_metrics_from_confusion_matrix(cm_df):
    """Calculate metrics from confusion matrix"""
    # Convert to numpy array (skip first row and column which are labels)
    cm = cm_df.iloc[1:, 1:].values.astype(float)
    
    # Calculate class-wise metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Calculate weighted averages
    class_support = np.sum(cm, axis=1)
    total_samples = np.sum(cm)
    weighted_precision = np.sum(precision * class_support) / total_samples
    weighted_recall = np.sum(recall * class_support) / total_samples
    weighted_f1 = np.sum(f1 * class_support) / total_samples
    
    return {
        'accuracy': np.trace(cm) / np.sum(cm),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }

def load_model_results():
    """Load results from confusion matrices"""
    results = {}
    
    # Logistic Regression
    lg_cm_path = Path("/Users/tianchuhang/Downloads/neisscode/model_outputs/logistic_regression/lg_confusion_matrix.csv")
    if lg_cm_path.exists():
        lg_cm_df = pd.read_csv(lg_cm_path, index_col=0)
        lg_metrics = calculate_metrics_from_confusion_matrix(lg_cm_df)
        results['Logistic Regression'] = lg_metrics
    
    # Random Forest
    rf_cm_path = Path("/Users/tianchuhang/Downloads/neisscode/model_outputs/random_forest/random_forest_confusion_matrix.csv")
    if rf_cm_path.exists():
        rf_cm_df = pd.read_csv(rf_cm_path, index_col=0)
        rf_metrics = calculate_metrics_from_confusion_matrix(rf_cm_df)
        results['Random Forest'] = rf_metrics
    
    # XGBoost
    xgb_cm_path = Path("/Users/tianchuhang/Downloads/neisscode/model_outputs/xgboost/xgboost_confusion_matrix.csv")
    if xgb_cm_path.exists():
        xgb_cm_df = pd.read_csv(xgb_cm_path, index_col=0)
        xgb_metrics = calculate_metrics_from_confusion_matrix(xgb_cm_df)
        results['XGBoost'] = xgb_metrics
    
    return results

def create_performance_table():
    """Create Table 2 with performance metrics"""
    results = load_model_results()
    
    # Create DataFrame
    data = []
    for model_name, metrics in results.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Macro Precision': f"{metrics['macro_precision']:.3f}",
            'Macro Recall': f"{metrics['macro_recall']:.3f}",
            'Macro F1': f"{metrics['macro_f1']:.3f}",
            'Weighted Precision': f"{metrics['weighted_precision']:.3f}",
            'Weighted Recall': f"{metrics['weighted_recall']:.3f}",
            'Weighted F1': f"{metrics['weighted_f1']:.3f}",
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def main():
    """Generate Table 2"""
    print("Generating Table 2: Performance metrics comparison...")
    
    # Create output directory
    output_dir = Path("/Users/tianchuhang/Downloads/neisscode/results/table1_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate table
    table2_df = create_performance_table()
    
    # Save to CSV
    output_path = output_dir / "table2_model_performance_simple.csv"
    table2_df.to_csv(output_path, index=False)
    
    print(f"Table 2 saved to: {output_path}")
    print("\nTABLE 2. Performance metrics on the models")
    print("=" * 80)
    print(table2_df.to_string(index=False))
    
    print(f"\nTable saved as CSV: {output_path}")

if __name__ == "__main__":
    main()
