import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, classification_report, confusion_matrix)
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import make_classification

def plot_roc_curve(y_true, y_prob, model_name="PhageAI Classifier", save_path=None, figsize=(8, 6)):
    """
    Generate and plot ROC curve for binary classification
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        model_name: Name of the model for the plot title
        save_path: Path to save the plot (optional)
        figsize: Figure size tuple
    
    Returns:
        tuple: (fpr, tpr, auc_score, fig, ax)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8')  # Modern style
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='#2E86AB', linewidth=3, 
             label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='#F24236', linestyle='--', linewidth=2,
             label='Random Classifier (AUC = 0.500)')
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add performance text box
    textstr = f'AUC Score: {roc_auc:.3f}\n'
    textstr += f'Model Performance: {"Excellent" if roc_auc > 0.9 else "Good" if roc_auc > 0.8 else "Fair" if roc_auc > 0.7 else "Poor"}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.6, 0.2, textstr, fontsize=10, bbox=props)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return fpr, tpr, roc_auc, plt.gcf(), plt.gca()

def enhanced_train_evaluate_model_with_roc(X, y, metadata=None, is_categorical=None, 
                                         hyperparameter_tuning=False, feature_selection=True,
                                         plot_roc=True, save_roc_path=None):
    """
    Enhanced version of your train_evaluate_model function that includes ROC curve plotting
    
    This is a modified version of your existing function with ROC curve generation added
    """
    
    print("Training and evaluating model with ROC curve...")
    
    # Determine if we should use classification or regression
    if is_categorical is None:
        if y is not None and (len(np.unique(y)) < 10 or isinstance(y.iloc[0], (str, bool))):
            is_categorical = True
        else:
            is_categorical = False
    
    model_type = "classification" if is_categorical else "regression"
    print(f"Detected problem type: {model_type}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, 
                                                        stratify=y if is_categorical else None)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Feature selection if requested
    if feature_selection and X_train.shape[1] > 20:
        print(f"Performing feature selection from {X_train.shape[1]} initial features")
        
        if is_categorical:
            selector = SelectFromModel(LGBMClassifier(n_estimators=100, random_state=42), threshold='median')
        else:
            selector = SelectFromModel(LGBMRegressor(n_estimators=100, random_state=42), threshold='median')
        
        selector.fit(X_train_scaled, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        print(f"Selected {len(selected_features)} important features")
        
        X_train_scaled = selector.transform(X_train_scaled)
        X_test_scaled = selector.transform(X_test_scaled)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)
    
    # Train the model
    if is_categorical:
        print("Training LightGBM classification model...")
        
        if hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 7, 9],
                'num_leaves': [15, 31, 63]
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model = GridSearchCV(LGBMClassifier(random_state=42, n_jobs=-1), param_grid,
                               cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1)
            model.fit(X_train_scaled, y_train)
            print(f"Best parameters: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=7,
                                 num_leaves=31, reg_alpha=0.1, reg_lambda=0.1,
                                 random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
    else:
        print("Training LightGBM regression model...")
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7,
                            num_leaves=31, reg_alpha=0.1, reg_lambda=0.1,
                            random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    print("Model Performance Metrics:")
    if is_categorical:
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Get probabilities for ROC curve (binary classification)
        if len(np.unique(y)) == 2:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate additional metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"AUC-ROC: {auc_roc:.4f}")
            
            # Plot ROC curve if requested
            if plot_roc:
                print("Generating ROC curve...")
                fpr, tpr, roc_auc, fig, ax = plot_roc_curve(
                    y_test, y_pred_proba, 
                    model_name="PhageAI Classifier",
                    save_path=save_roc_path
                )
                return model, scaler, None, is_categorical, (fpr, tpr, roc_auc, fig, ax)
        else:
            # Multi-class classification
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"Precision (weighted): {precision:.4f}")
            print(f"Recall (weighted): {recall:.4f}")
            print(f"F1 Score (weighted): {f1:.4f}")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_columns = X_train_scaled.columns.tolist()
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        for idx, (feature, importance) in enumerate(zip(feature_importance['Feature'].head(10), 
                                                       feature_importance['Importance'].head(10))):
            print(f"{idx:3d} {feature:30s} {importance:.4f}")
    else:
        feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
    
    return model, scaler, feature_importance, is_categorical

def plot_multi_model_roc(models_data, save_path=None, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models on the same plot
    
    Args:
        models_data: List of tuples (model_name, y_true, y_prob)
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#593E2C']
    
    for i, (model_name, y_true, y_prob) in enumerate(models_data):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        color = colors[i % len(colors)]
        plt.plot(fpr, tpr, color=color, linewidth=2.5, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
             label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-model ROC curve saved to {save_path}")
    
    plt.show()

def generate_roc_curve_for_phage_model(data_path, target_column=None, save_path="roc_curve.png"):
    """
    Complete pipeline to generate ROC curve for your phage model
    
    Args:
        data_path: Path to your dataset
        target_column: Name of target column
        save_path: Where to save the ROC curve plot
    """
    # This would use your existing functions
    # data = load_and_explore_data(data_path)
    # X, y, metadata, column_groups, is_categorical, feature_columns, target_column = preprocess_data(data, target_column)
    # model, scaler, feature_importance, is_categorical, roc_data = enhanced_train_evaluate_model_with_roc(
    #     X, y, metadata, is_categorical, plot_roc=True, save_roc_path=save_path
    # )
    
    print(f"ROC curve analysis complete. Plot saved to {save_path}")
    return # model, scaler, feature_importance, roc_data

# DEMO EXECUTION - This will run when you execute the script
if __name__ == "__main__":
    print("=" * 60)
    print("ROC CURVE DEMO - PHAGE AI CLASSIFIER")
    print("=" * 60)
    
    # Generate sample binary classification data for demonstration
    print("Generating sample dataset...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=25, 
        n_classes=2, 
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Convert to DataFrame and Series (like your real data)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Run the enhanced model training with ROC curve
    print("\nTraining model and generating ROC curve...")
    result = enhanced_train_evaluate_model_with_roc(
        X_df, y_series,
        hyperparameter_tuning=False,  # Set to True for better performance (takes longer)
        feature_selection=True,
        plot_roc=True,
        save_roc_path="phage_classifier_roc.png"
    )
    
    # Unpack results
    if len(result) == 5:
        model, scaler, feature_importance, is_categorical, roc_data = result
        fpr, tpr, auc_score, fig, ax = roc_data
        print(f"\nROC Analysis Complete!")
        print(f"Final AUC Score: {auc_score:.4f}")
        print(f"ROC curve saved as: phage_classifier_roc.png")
    else:
        model, scaler, feature_importance, is_categorical = result
        print("\nModel training complete (no ROC curve for multi-class)")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE - Check for 'phage_classifier_roc.png' file")
    print("=" * 60)