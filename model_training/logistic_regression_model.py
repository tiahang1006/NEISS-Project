import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTDIR = Path("/Users/tianchuhang/Downloads/neisscode/lg_output")
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare_data():
    """
    Load the cleaned data and prepare for modeling
    """
    print("Loading cleaned NEISS data...")
    df = pd.read_csv('neiss_cleaned_2016_2024.csv')
    print(f"Data loaded: {df.shape}")
    return df

def encode_categorical_variables(df):
    """
    Apply one-hot encoding to categorical variables
    """
    print("Applying one-hot encoding to categorical variables...")
    
    # Define categorical columns (excluding target)
    categorical_columns = ['Sex', 'Race', 'Age_Category', 'Diagnosis_Category', 
                          'Product_Category', 'Season', 'Location', 'Disposition', 'Fire_Involvement']
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)
    
    print(f"Shape after encoding: {df_encoded.shape}")
    print(f"Number of features: {df_encoded.shape[1] - 1}")  # -1 for target column
    
    return df_encoded

def prepare_model_data(df_encoded, target_col="Body_Part_Category"):
    """
    Prepare data for modeling
    """
    print(f"Preparing data for modeling with target: {target_col}")
    
    # Separate features and target
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    return X, y

def train_logistic_regression(X, y, test_size=0.2, random_state=42):
    """
    Train logistic regression model with improved settings
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train logistic regression model with improved settings
    print("\nTraining logistic regression model...")
    lr_model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        multi_class='multinomial',  # Better for multiclass
        solver='lbfgs'  # Better for multinomial
    )
    
    # No standardization needed for one-hot encoded features
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    y_pred_proba = lr_model.predict_proba(X_test)
    
    print("Model training completed!")
    
    return lr_model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba, lr_model):
    """
    Evaluate the model performance with enhanced metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Get class names from model
    class_names = lr_model.classes_
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Enhanced Classification Report with macro and weighted F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix with consistent class ordering
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    print(cm)
    
    # Save confusion matrix to CSV with consistent ordering
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(OUTDIR / 'lg_confusion_matrix.csv')
    print(f"Confusion matrix saved to: {OUTDIR / 'lg_confusion_matrix.csv'}")
    
    # Multi-class ROC analysis
    print("\nMulti-class ROC Analysis:")
    try:
        from sklearn.metrics import roc_auc_score
        # For multiclass, we can calculate macro and weighted ROC AUC
        roc_auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        roc_auc_weighted = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        print(f"Macro ROC AUC: {roc_auc_macro:.4f}")
        print(f"Weighted ROC AUC: {roc_auc_weighted:.4f}")
    except Exception as e:
        print(f"ROC analysis skipped: {e}")
    
    return accuracy, cm, macro_f1, weighted_f1

def analyze_feature_importance(lr_model, feature_names):
    """
    Analyze feature importance (coefficients) with consistent class ordering
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get class names from model for consistent ordering
    class_names = lr_model.classes_
    
    # Get coefficients for each class
    coefficients = lr_model.coef_
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Number of features: {len(feature_names)}")
    
    # Create feature importance DataFrame with consistent class ordering
    feature_importance_df = pd.DataFrame(
        coefficients.T,
        index=feature_names,
        columns=class_names
    )
    
    # Calculate absolute mean importance across all classes
    feature_importance_df['Mean_Abs_Importance'] = np.abs(feature_importance_df).mean(axis=1)
    feature_importance_df = feature_importance_df.sort_values('Mean_Abs_Importance', ascending=False)
    
    print("\nTop 20 Most Important Features (by absolute mean coefficient):")
    print(feature_importance_df.head(20)[['Mean_Abs_Importance']])
    
    # Save feature importance to CSV
    feature_importance_df.to_csv(OUTDIR / 'lg_feature_importance.csv')
    print(f"Feature importance saved to: {OUTDIR / 'lg_feature_importance.csv'}")
    
    return feature_importance_df

def generate_model_summary(lr_model, feature_names, accuracy, cm, macro_f1, weighted_f1):
    """
    Generate comprehensive model summary with enhanced metrics
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY AND INTERPRETATION")
    print("="*60)
    
    # Get class names from model
    class_names = lr_model.classes_
    
    print("1. MODEL PERFORMANCE:")
    print(f"   - Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - Macro F1-score: {macro_f1:.4f}")
    print(f"   - Weighted F1-score: {weighted_f1:.4f}")
    print(f"   - Number of classes: {len(class_names)}")
    print(f"   - Number of features: {len(feature_names)}")
    
    print("\n2. CLASS DISTRIBUTION IN CONFUSION MATRIX:")
    for i, class_name in enumerate(class_names):
        true_positives = cm[i, i]
        total_actual = cm[i, :].sum()
        recall = true_positives / total_actual if total_actual > 0 else 0
        print(f"   - {class_name}: {true_positives}/{total_actual} ({recall:.3f})")
    
    print("\n3. MODEL INTERPRETATION:")
    print("   - This is a multiclass logistic regression model")
    print("   - It uses multinomial strategy for multiclass classification")
    print("   - Each class has its own set of coefficients")
    print("   - Positive coefficients increase the probability of that class")
    print("   - Negative coefficients decrease the probability of that class")
    
    print("\n4. HOW TO USE THE MODEL:")
    print("   - Input: Patient characteristics (demographics, diagnosis, etc.)")
    print("   - Output: Predicted body part category")
    print("   - The model provides probabilities for each body part category")
    print("   - The class with highest probability is the predicted category")
    
    print("\n5. FEATURE IMPORTANCE:")
    print("   - Features with larger absolute coefficients are more important")
    print("   - Positive coefficients favor the class")
    print("   - Negative coefficients disfavor the class")
    
    print("\n6. IMPROVEMENTS MADE:")
    print("   - Removed unnecessary standardization for one-hot features")
    print("   - Used multinomial strategy instead of OvR")
    print("   - Added macro and weighted F1-scores")
    print("   - Consistent class ordering throughout")
    print("   - Enhanced ROC analysis for multiclass")
    
    return True

def save_model_results(lr_model, feature_names, accuracy, macro_f1, weighted_f1):
    """
    Save model and results for future use
    """
    print("\n" + "="*60)
    print("SAVING MODEL RESULTS")
    print("="*60)
    
    # Get class names from model
    class_names = lr_model.classes_
    
    # Save model information
    model_info = {
        'model_type': 'Logistic Regression (Improved)',
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'n_features': len(feature_names),
        'n_classes': len(class_names),
        'class_names': list(class_names),
        'feature_names': feature_names
    }
    
    # Save to CSV
    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_csv(OUTDIR / 'lg_model_info.csv', index=False)
    
    print(f"Model information saved to: {OUTDIR / 'lg_model_info.csv'}")
    print(f"Confusion matrix saved to: {OUTDIR / 'lg_confusion_matrix.csv'}")
    print(f"Feature importance saved to: {OUTDIR / 'lg_feature_importance.csv'}")
    
    return True

def main():
    """
    Main function to run the complete modeling pipeline
    """
    print("LOGISTIC REGRESSION MODELING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    df = load_and_prepare_data()
    
    # Step 2: Encode categorical variables
    df_encoded = encode_categorical_variables(df)
    
    # Step 3: Prepare model data
    X, y = prepare_model_data(df_encoded, target_col="Body_Part_Category")
    
    # Step 4: Train model
    lr_model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba = train_logistic_regression(X, y)
    
    # Step 5: Evaluate model
    accuracy, cm, macro_f1, weighted_f1 = evaluate_model(y_test, y_pred, y_pred_proba, lr_model)
    
    # Step 6: Analyze feature importance
    feature_names = X.columns.tolist()
    feature_importance_df = analyze_feature_importance(lr_model, feature_names)
    
    # Step 7: Generate model summary
    generate_model_summary(lr_model, feature_names, accuracy, cm, macro_f1, weighted_f1)
    
    # Step 8: Save results
    save_model_results(lr_model, feature_names, accuracy, macro_f1, weighted_f1)
    
    print("\n" + "="*60)
    print("MODELING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files generated:")
    print(f"- {OUTDIR / 'lg_model_info.csv'}: Model performance metrics")
    print(f"- {OUTDIR / 'lg_confusion_matrix.csv'}: Confusion matrix data")
    print(f"- {OUTDIR / 'lg_feature_importance.csv'}: Feature importance data")
    
    return lr_model, feature_names, accuracy

if __name__ == "__main__":
    model, features, acc = main()
