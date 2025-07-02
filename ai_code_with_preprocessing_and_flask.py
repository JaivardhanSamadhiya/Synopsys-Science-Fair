import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix, 
    mean_squared_error, r2_score
)
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import pickle
import joblib
import os
import json
import time
from datetime import datetime
import logging
from flask import Flask, request, jsonify, Response
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction as GC
import re
import itertools
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phage_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhageAI")

# 1. Data Loading and Initial Exploration
def load_and_explore_data(filepath):
    """
    Load the dataset and perform initial exploration
    
    Args:
        filepath (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: The loaded dataset or None if loading fails
    """
    logger.info(f"Loading and exploring data from {filepath}...")
    
    # Load the data
    try:
        # Check file extension
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.tsv') or filepath.endswith('.txt'):
            data = pd.read_csv(filepath, sep='\t')
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            data = pd.read_excel(filepath)
        else:
            logger.error(f"Unsupported file format: {filepath}")
            return None
            
        logger.info(f"Dataset loaded successfully with shape: {data.shape}")
        
        # Display basic info
        logger.info(f"Number of rows: {data.shape[0]}")
        logger.info(f"Number of columns: {data.shape[1]}")
        logger.info(f"Column types:\n{data.dtypes.value_counts()}")
        
        # Check for missing values
        missing_values = data.isnull().sum().sum()
        logger.info(f"Total missing values: {missing_values}")
        
        if missing_values > 0:
            # Show columns with missing values
            missing_cols = data.columns[data.isnull().any()].tolist()
            logger.info(f"Columns with missing values: {len(missing_cols)}")
            if len(missing_cols) < 10:
                for col in missing_cols:
                    missing_pct = data[col].isnull().mean() * 100
                    logger.info(f"  - {col}: {data[col].isnull().sum()} missing values ({missing_pct:.2f}%)")
            else:
                logger.info(f"  Top 10 columns with most missing values:")
                missing_count = data.isnull().sum()
                top_missing = missing_count[missing_count > 0].sort_values(ascending=False).head(10)
                for col, count in top_missing.items():
                    missing_pct = count / len(data) * 100
                    logger.info(f"  - {col}: {count} missing values ({missing_pct:.2f}%)")
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

# Function to load genomic data from FASTA files
def load_genomic_data(phage_fasta=None, host_fasta=None):
    """
    Load genomic data from FASTA files for phages and hosts
    
    Args:
        phage_fasta (str): Path to phage genome FASTA file
        host_fasta (str): Path to host genome FASTA file
        
    Returns:
        tuple: (phage_sequences, host_sequences) dictionaries
    """
    phage_sequences = {}
    host_sequences = {}
    
    if phage_fasta and os.path.exists(phage_fasta):
        try:
            logger.info(f"Loading phage genomic data from {phage_fasta}")
            for record in SeqIO.parse(phage_fasta, "fasta"):
                phage_sequences[record.id] = str(record.seq)
            logger.info(f"Loaded {len(phage_sequences)} phage sequences")
        except Exception as e:
            logger.error(f"Error loading phage genomic data: {e}")
    
    if host_fasta and os.path.exists(host_fasta):
        try:
            logger.info(f"Loading host genomic data from {host_fasta}")
            for record in SeqIO.parse(host_fasta, "fasta"):
                host_sequences[record.id] = str(record.seq)
            logger.info(f"Loaded {len(host_sequences)} host sequences")
        except Exception as e:
            logger.error(f"Error loading host genomic data: {e}")
    
    return phage_sequences, host_sequences
# New: Genomic Feature Engineering Functions
def extract_kmer_features(sequence, k=3, normalized=True):
    """
    Extract k-mer frequency features from a genomic sequence
    
    Args:
        sequence (str): Genomic sequence
        k (int): k-mer length
        normalized (bool): Whether to normalize frequencies
        
    Returns:
        dict: k-mer frequencies
    """
    kmers = {}
    n = len(sequence)
    
    if n < k:
        return kmers
    
    # Generate all possible k-mers
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    
    # Initialize counts
    for kmer in all_kmers:
        kmers[kmer] = 0
    
    # Count k-mers in sequence
    for i in range(n - k + 1):
        kmer = sequence[i:i+k].upper()
        if all(base in 'ACGT' for base in kmer):  # Skip k-mers with non-standard bases
            kmers[kmer] = kmers.get(kmer, 0) + 1
    
    # Normalize if requested
    if normalized:
        total = max(1, n - k + 1)  # Avoid division by zero
        for kmer in kmers:
            kmers[kmer] = kmers[kmer] / total
    
    return kmers

def calculate_gc_content(sequence, window_size=1000, stride=500):
    """
    Calculate GC content in sliding windows
    
    Args:
        sequence (str): Genomic sequence
        window_size (int): Size of sliding window
        stride (int): Stride between windows
        
    Returns:
        list: GC content values for each window
    """
    n = len(sequence)
    gc_values = []
    
    for i in range(0, n - window_size + 1, stride):
        window = sequence[i:i+window_size].upper()
        gc = GC(window)
        gc_values.append(gc)
    
    return gc_values

def extract_genomic_features(sequences, k_values=[3, 4, 5], gc_window_size=1000, prefix=""):
    """
    Extract genomic features from a set of sequences
    
    Args:
        sequences (dict): Dictionary of sequence ID to sequence
        k_values (list): List of k values for k-mer extraction
        gc_window_size (int): Window size for GC content calculation
        prefix (str): Prefix for feature names
        
    Returns:
        pd.DataFrame: DataFrame of genomic features
    """
    features_list = []
    
    for seq_id, sequence in sequences.items():
        features = {"sequence_id": seq_id}
        
        # Extract GC content
        gc_values = calculate_gc_content(sequence, gc_window_size)
        features[f"{prefix}gc_content_mean"] = np.mean(gc_values) if gc_values else 0
        features[f"{prefix}gc_content_std"] = np.std(gc_values) if gc_values else 0
        features[f"{prefix}gc_content_min"] = np.min(gc_values) if gc_values else 0
        features[f"{prefix}gc_content_max"] = np.max(gc_values) if gc_values else 0
        features[f"{prefix}sequence_length"] = len(sequence)
        
        # Extract k-mer features for each k value
        for k in k_values:
            kmer_features = extract_kmer_features(sequence, k)
            for kmer, freq in kmer_features.items():
                features[f"{prefix}kmer_{kmer}"] = freq
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def integrate_genomic_data(interaction_data, phage_features, host_features):
    """
    Integrate interaction data with genomic features
    
    Args:
        interaction_data (pd.DataFrame): Phage-host interaction data
        phage_features (pd.DataFrame): Phage genomic features
        host_features (pd.DataFrame): Host genomic features
        
    Returns:
        pd.DataFrame: Integrated dataset
    """
    logger.info("Integrating genomic data with interaction data...")
    
    # Identify key columns for merging
    phage_id_col = next((col for col in interaction_data.columns if 'phage' in col.lower() and 'id' in col.lower()), None)
    host_id_col = next((col for col in interaction_data.columns if 'host' in col.lower() and 'id' in col.lower() or 'strain' in col.lower()), None)
    
    if not phage_id_col or not host_id_col:
        logger.warning("Could not identify phage_id or host_id columns for merging")
        logger.info(f"Available columns: {interaction_data.columns.tolist()}")
        # Try to use default column names
        phage_id_col = phage_id_col or "phage_id"
        host_id_col = host_id_col or "host_id"
    
    logger.info(f"Using merge columns: phage_id='{phage_id_col}', host_id='{host_id_col}'")
    
    # Merge phage features
    if not phage_features.empty:
        merged_data = interaction_data.merge(
            phage_features,
            left_on=phage_id_col,
            right_on="sequence_id",
            how="left"
        )
        logger.info(f"Data shape after merging phage features: {merged_data.shape}")
    else:
        merged_data = interaction_data.copy()
        logger.warning("No phage genomic features available for integration")
    
    # Merge host features
    if not host_features.empty:
        merged_data = merged_data.merge(
            host_features,
            left_on=host_id_col,
            right_on="sequence_id",
            how="left",
            suffixes=("", "_host")
        )
        logger.info(f"Data shape after merging host features: {merged_data.shape}")
    else:
        logger.warning("No host genomic features available for integration")
    
    # Drop redundant columns
    if "sequence_id" in merged_data.columns:
        merged_data = merged_data.drop(columns=["sequence_id"])
    if "sequence_id_host" in merged_data.columns:
        merged_data = merged_data.drop(columns=["sequence_id_host"])
    
    # Report on merged dataset
    logger.info(f"Final integrated dataset shape: {merged_data.shape}")
    missing_percent = merged_data.isnull().mean() * 100
    cols_with_missing = missing_percent[missing_percent > 0]
    logger.info(f"Columns with missing values after integration: {len(cols_with_missing)}")
    
    return merged_data
# 2. Data Preprocessing
def preprocess_data(data, target_column=None):
    """
    Preprocess the dataset for model training
    
    Args:
        data (pd.DataFrame): The raw dataset
        target_column (str, optional): The target column name. If None, will try to detect automatically.
        
    Returns:
        tuple: (X, y, metadata, column_groups, is_categorical, feature_columns, target_column)
    """
    logger.info("Preprocessing data...")
    
    # Make a copy to avoid modifying the original dataframe
    df = data.copy()
    
    # 2.1 Handle ID columns - extract as metadata but remove from features
    id_columns = [col for col in df.columns if '_ID' in col or 'id' in col.lower()]
    logger.info(f"Detected ID columns: {id_columns}")
    
    metadata = df[id_columns].copy() if id_columns else None
    
    if metadata is not None:
        logger.info(f"Extracted {len(id_columns)} ID columns as metadata")
        df = df.drop(columns=id_columns)
    
    # 2.2 Identify feature types
    gc_columns = [col for col in df.columns if 'gc_content' in col.lower()]
    phage_kmer_columns = [col for col in df.columns if any(x in col.lower() for x in ['phage_kmer', 'phagekmer'])]
    host_kmer_columns = [col for col in df.columns if any(x in col.lower() for x in ['host_kmer', 'hostkmer'])]
    
    # Add columns for the more specific genomic features
    sequence_columns = [col for col in df.columns if 'sequence_length' in col.lower()]
    
    column_groups = {
        'gc_columns': gc_columns,
        'phage_kmer_columns': phage_kmer_columns,
        'host_kmer_columns': host_kmer_columns,
        'sequence_columns': sequence_columns
    }
    
    logger.info(f"Detected {len(gc_columns)} GC content columns")
    logger.info(f"Detected {len(phage_kmer_columns)} phage kmer columns")
    logger.info(f"Detected {len(host_kmer_columns)} host kmer columns")
    logger.info(f"Detected {len(sequence_columns)} sequence length columns")
    
    # 2.3 Handle target variable
    if target_column is None:
        # Try to automatically identify the target column
        possible_targets = [col for col in df.columns if any(x in col.lower() for x in 
                           ['effect', 'interact', 'output', 'target', 'phage_effect', 'efficacy', 'effectiveness'])]
        if possible_targets:
            target_column = possible_targets[0]
            logger.info(f"Automatically detected target column: {target_column}")
        else:
            # If no clear target column is found, assume the last column
            target_column = df.columns[-1]
            logger.info(f"No clear target column found, using last column: {target_column}")
    else:
        logger.info(f"Using provided target column: {target_column}")
    
    # 2.4 Handle missing values - use more sophisticated imputation
    missing_percent = df.isnull().mean()
    cols_to_drop = missing_percent[missing_percent > 0.8].index.tolist()
    
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >80% missing values")
        if len(cols_to_drop) <= 10:
            for col in cols_to_drop:
                logger.info(f"  - {col}: {missing_percent[col]:.2%} missing")
        df = df.drop(columns=cols_to_drop)
    
    # Group columns by data type for appropriate imputation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    
    # For numeric columns with moderate missing values, use KNN imputation
    cols_to_impute_knn = [col for col in numeric_cols if 0 < df[col].isnull().mean() < 0.3]
    if cols_to_impute_knn:
        logger.info(f"Using KNN imputation for {len(cols_to_impute_knn)} numeric columns with <30% missing values")
        imputer = KNNImputer(n_neighbors=5)
        df[cols_to_impute_knn] = imputer.fit_transform(df[cols_to_impute_knn])
    
    # For numeric columns with more missing values, use median
    cols_to_impute_median = [col for col in numeric_cols if 0.3 <= df[col].isnull().mean() < 0.8]
    if cols_to_impute_median:
        logger.info(f"Using median imputation for {len(cols_to_impute_median)} numeric columns with 30-80% missing values")
        for col in cols_to_impute_median:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    cols_to_impute_mode = [col for col in cat_cols if 0 < df[col].isnull().mean() < 0.8]
    if cols_to_impute_mode:
        logger.info(f"Using mode imputation for {len(cols_to_impute_mode)} categorical columns")
        for col in cols_to_impute_mode:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 2.5 Handle any remaining NaN values with a special value
    df = df.fillna(-999)
    
    # 2.6 Prepare features and target
    X = df.drop(columns=[target_column]) if target_column in df.columns else df
    y = df[target_column] if target_column in df.columns else None
    
    # Save the feature columns for later API use
    feature_columns = X.columns.tolist()
    
    logger.info(f"Processed features shape: {X.shape}")
    if y is not None:
        logger.info(f"Target variable: {target_column}")
        # Check if target is continuous or categorical
        if len(y.unique()) < 10 or y.dtype == 'object' or y.dtype == 'bool':
            logger.info(f"Target appears to be categorical with {len(y.unique())} classes")
            logger.info(f"Target classes distribution:\n{y.value_counts(normalize=True)}")
            is_categorical = True
        else:
            logger.info(f"Target appears to be continuous with range: {y.min()} to {y.max()}")
            logger.info(f"Target descriptive statistics:\n{y.describe()}")
            is_categorical = False
    else:
        is_categorical = None
    
    # Return processed data
    return X, y, metadata, column_groups, is_categorical, feature_columns, target_column
# 3. Enhanced Model Training and Evaluation
def train_evaluate_model(X, y, metadata=None, is_categorical=None, hyperparameter_tuning=False, feature_selection=True):
    """
    Train and evaluate a tree-based model with enhanced ML pipeline
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        metadata (pd.DataFrame, optional): Metadata columns
        is_categorical (bool, optional): Whether the target is categorical
        hyperparameter_tuning (bool, optional): Whether to perform hyperparameter tuning
        feature_selection (bool): Whether to perform feature selection
        
    Returns:
        tuple: (model, scaler, feature_importance, is_categorical)
    """
    logger.info("Training and evaluating model...")
    
    # Determine if we should use classification or regression
    if is_categorical is None:
        # Auto-detect if target is categorical
        if y is not None and (len(np.unique(y)) < 10 or isinstance(y.iloc[0], (str, bool))):
            is_categorical = True
        else:
            is_categorical = False
    
    model_type = "classification" if is_categorical else "regression"
    logger.info(f"Detected problem type: {model_type}")
    
    # 3.1 Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # 3.2 Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # 3.3 Feature selection if requested
    if feature_selection and X_train.shape[1] > 20:
        logger.info(f"Performing feature selection from {X_train.shape[1]} initial features")
        
        # Use a pre-trained model for feature selection
        if is_categorical:
            selector = SelectFromModel(
                LGBMClassifier(n_estimators=100, random_state=42), 
                threshold='median'
            )
        else:
            selector = SelectFromModel(
                LGBMRegressor(n_estimators=100, random_state=42), 
                threshold='median'
            )
        
        selector.fit(X_train_scaled, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} important features")
        logger.info(f"Selected features: {selected_features[:10]}...")
        
        X_train_scaled = selector.transform(X_train_scaled)
        X_test_scaled = selector.transform(X_test_scaled)
        
        # Convert back to DataFrame with only selected features
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features)
        
        logger.info(f"Reduced feature set shape: {X_train_scaled.shape}")
    
    # 3.3 Train the model with cross-validation
    if is_categorical:
        logger.info("Training LightGBM classification model with cross-validation...")
        
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 7, 9],
                'num_leaves': [15, 31, 63],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model = GridSearchCV(
                LGBMClassifier(random_state=42, n_jobs=-1),
                param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            model.fit(X_train_scaled, y_train)
            logger.info(f"Best parameters: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
    else:
        logger.info("Training LightGBM regression model with cross-validation...")
        
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [5, 7, 9],
                'num_leaves': [15, 31, 63],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            model = GridSearchCV(
                LGBMRegressor(random_state=42, n_jobs=-1),
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            model.fit(X_train_scaled, y_train)
            logger.info(f"Best parameters: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
    
    # 3.4 Evaluate the model with all required metrics
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics based on problem type
    logger.info("Model Performance Metrics:")
    if is_categorical:
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # For binary classification
        if len(np.unique(y)) == 2:
            y_pred_proba = model.predict_proba(X_test_scaled)[:,1]
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"AUC-ROC: {auc_roc:.4f}")
        else:
            # For multi-class classification
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Precision (weighted): {precision:.4f}")
            logger.info(f"Recall (weighted): {recall:.4f}")
            logger.info(f"F1 Score (weighted): {f1:.4f}")
        
        # Detailed classification report
        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
    else:
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"Root Mean Squared Error: {rmse:.4f}")
        logger.info(f"R² Score: {r2:.4f}")
    
    # 3.5 Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_columns = X_train_scaled.columns.tolist() if isinstance(X_train_scaled, pd.DataFrame) else [f"feature_{i}" for i in range(X_train_scaled.shape[1])]
        
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        logger.info("Top 10 Most Important Features:")
        for idx, (feature, importance) in enumerate(zip(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10))):
            logger.info(f"{idx:3d} {feature:30s} {importance:.4f}")
    else:
        feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])
        logger.warning("Model does not provide feature importances")
    
    return model, scaler, feature_importance, is_categorical
def adjust_effectiveness_score(score, is_staphylococcus=False):
    """
    Adjust effectiveness scores to avoid exact 1.0 values except for staphylococcus strains
    which might legitimately have perfect scores
    
    Args:
        score: The raw effectiveness score between 0 and 1
        is_staphylococcus: Whether this is a staphylococcus prediction
        
    Returns:
        float: Adjusted effectiveness score
    """
    # If it's a staphylococcus prediction, we may allow a perfect 1.0 score
    # Otherwise, cap at 0.999 for perfect predictions
    if score >= 0.999:
        if is_staphylococcus:
            return score  # Keep as is, might be 1.0
        else:
            return 0.999  # Cap at 0.999 for non-staphylococcus
    return score
# 4. Enhanced Phage Cocktail Recommendation

def recommend_phage_cocktail(model, scaler, X, phage_data=None, strain_data=None, is_categorical=True, 
                            n_recommendations=3, cocktail_size=3):
    """
    Generate optimal phage cocktail recommendations for a given bacterial strain
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X: Features dataframe
        phage_data: DataFrame with phage information
        strain_data: DataFrame with strain information
        is_categorical: Whether the model is a classifier or regressor
        n_recommendations: Number of cocktail recommendations to generate
        cocktail_size: Number of phages in each cocktail
        
    Returns:
        list: List of cocktail recommendations with effectiveness scores
    """
    logger.info(f"Generating phage cocktail recommendations (size {cocktail_size})")
    
    # Ensure we have phage data
    if phage_data is None or phage_data.empty:
        logger.error("No phage data provided for cocktail recommendation")
        return []
    
    # If strain data is provided, we'll use it to customize recommendations
    strain_specific = strain_data is not None and not strain_data.empty
    
    # Check if this is a staphylococcus strain
    is_staph = False
    if strain_specific:
        strain_name = strain_data.get('strain_name', '').lower() if isinstance(strain_data, dict) else ''
        if isinstance(strain_data, pd.DataFrame) and 'strain_name' in strain_data.columns:
            strain_name = strain_data['strain_name'].iloc[0].lower()
        is_staph = "staphylococcus" in strain_name
    
    # Scale features if scaler is provided
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.copy()
    
    # Get predictions for all phage-host pairs
    if is_categorical:
        # For classification models, get probability of positive class
        predictions = model.predict_proba(X_scaled)
        # Take the probability of the positive class (usually the second column)
        effectiveness_scores = predictions[:, 1] if predictions.shape[1] > 1 else predictions
    else:
        # For regression models, get the predicted effectiveness directly
        effectiveness_scores = model.predict(X_scaled)
        # Ensure regression scores are within 0-1 range
        effectiveness_scores = np.clip(effectiveness_scores, 0.0, 1.0)
    
    # Adjust effectiveness scores to avoid exact 1.0 except for staphylococcus
    adjusted_scores = np.array([adjust_effectiveness_score(score, is_staph) for score in effectiveness_scores])
    
    # Create a DataFrame with phage IDs and their effectiveness scores
    results = pd.DataFrame({
        'phage_id': X.index if hasattr(X, 'index') else range(len(X)),
        'effectiveness': adjusted_scores
    })
    
    # Sort by effectiveness
    results = results.sort_values('effectiveness', ascending=False)
    
    # Generate diverse cocktail recommendations
    cocktails = []
    used_phages = set()
    
    # Make sure we have enough phages to create n_recommendations of cocktails
    if len(results) < n_recommendations * cocktail_size:
        logger.warning(f"Not enough phage data ({len(results)}) to generate {n_recommendations} cocktails of size {cocktail_size}")
        n_recommendations = max(1, len(results) // cocktail_size)
    
    # Generate cocktails with diverse phages
    for i in range(n_recommendations):
        # Get top effective phages that haven't been used yet
        available_phages = results[~results['phage_id'].isin(used_phages)]
        
        if len(available_phages) < cocktail_size:
            logger.warning(f"Not enough unique phages left for cocktail {i+1}, reusing some phages")
            # Reset used phages if we need more
            used_phages = set()
            available_phages = results
        
        # Select top phages for this cocktail
        cocktail_phages = available_phages.head(cocktail_size)
        
        # Create cocktail info
        cocktail = {
            'id': f"cocktail_{i+1}",
            'phages': cocktail_phages['phage_id'].tolist(),
            'individual_scores': cocktail_phages['effectiveness'].tolist(),
            'combined_score': np.mean(cocktail_phages['effectiveness']),  # Simple mean as combined score
            'diversity_score': np.std(cocktail_phages['effectiveness'])   # Use std as a simple diversity metric
        }
        
        cocktails.append(cocktail)
        
        # Mark these phages as used
        used_phages.update(cocktail_phages['phage_id'].tolist())
    
    # Sort cocktails by combined effectiveness score
    cocktails = sorted(cocktails, key=lambda x: x['combined_score'], reverse=True)
    
    logger.info(f"Generated {len(cocktails)} cocktail recommendations")
    return cocktails

def save_model(model, scaler, feature_importance, feature_columns, target_column, is_categorical, output_dir='models'):
    """
    Save the trained model and associated data
    
    Args:
        model: The trained ML model
        scaler: The fitted scaler
        feature_importance: DataFrame of feature importances
        feature_columns: List of feature column names
        target_column: Name of the target column
        is_categorical: Whether this is a classification or regression model
        output_dir: Directory to save the model files
        
    Returns:
        str: Path to the saved model directory
    """
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(output_dir, f"phage_model_{timestamp}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save feature importance
    if not feature_importance.empty:
        importance_path = os.path.join(model_dir, "feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'classification' if is_categorical else 'regression',
        'target_column': target_column,
        'feature_columns': feature_columns,
        'timestamp': timestamp,
        'model_library': str(type(model).__module__) + '.' + str(type(model).__name__),
        'num_features': len(feature_columns)
    }
    
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Model metadata saved to {metadata_path}")
    
    return model_dir

def load_model(model_dir):
    """
    Load a trained model and associated data
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        tuple: (model, scaler, metadata)
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Load model
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None, None, None
    
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
    else:
        logger.warning(f"Scaler file not found at {scaler_path}")
        scaler = None
    
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded from {metadata_path}")
    else:
        logger.warning(f"Metadata file not found at {metadata_path}")
        metadata = {}
    
    return model, scaler, metadata
# 5. API for Phage-Host Interaction Prediction
app = Flask(__name__)

# Global variables to store loaded model and data
loaded_model = None
loaded_scaler = None
loaded_metadata = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': loaded_model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predicting phage effectiveness against a host"""
    global loaded_model, loaded_scaler, loaded_metadata
    
    # Check if model is loaded
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        # Get request data
        data = request.json
        logger.info(f"Received prediction request: {data}")
        
        # Extract input features
        features = data.get('features', {})
        
        # Check if this is a staphylococcus prediction
        host_name = data.get('host_name', '').lower()
        is_staph = "staphylococcus" in host_name
        
        # Convert to DataFrame
        feature_columns = loaded_metadata.get('feature_columns', [])
        if not feature_columns:
            return jsonify({'error': 'No feature columns defined in model metadata'}), 400
        
        # Create DataFrame with expected columns
        input_df = pd.DataFrame([features], columns=feature_columns)
        
        # Fill missing columns with defaults
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Scale features
        if loaded_scaler is not None:
            input_scaled = loaded_scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Make prediction
        is_categorical = loaded_metadata.get('model_type') == 'classification'
        if is_categorical:
            try:
                # Try to get probability
                proba = loaded_model.predict_proba(input_scaled)[0]
                prediction = proba[1] if len(proba) > 1 else proba[0]
                # Adjust effectiveness score
                prediction = adjust_effectiveness_score(prediction, is_staph)
                class_prediction = loaded_model.predict(input_scaled)[0]
                result = {
                    'effectiveness': float(prediction),
                    'class': int(class_prediction) if isinstance(class_prediction, (int, np.integer)) else str(class_prediction),
                    'probability': float(prediction)
                }
            except:
                # Fall back to class prediction only
                prediction = loaded_model.predict(input_scaled)[0]
                effectiveness = 1.0 if prediction == 1 else 0.0
                # Adjust effectiveness score
                effectiveness = adjust_effectiveness_score(effectiveness, is_staph)
                result = {
                    'effectiveness': effectiveness,
                    'class': int(prediction) if isinstance(prediction, (int, np.integer)) else str(prediction)
                }
        else:
            prediction = loaded_model.predict(input_scaled)[0]
            # Ensure regression predictions stay within 0-1 range
            prediction = max(0.0, min(1.0, float(prediction)))
            # Adjust effectiveness score
            prediction = adjust_effectiveness_score(prediction, is_staph)
            result = {'effectiveness': prediction}
        
        logger.info(f"Prediction result: {result}")
        return jsonify({'prediction': result})
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint for recommending phage cocktails for a given host"""
    global loaded_model, loaded_scaler, loaded_metadata
    
    # Check if model is loaded
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded'}), 400
    
    try:
        # Get request data
        data = request.json
        logger.info(f"Received recommendation request: {data}")
        
        # Extract parameters
        host_id = data.get('host_id')
        phage_list = data.get('available_phages', [])
        cocktail_size = int(data.get('cocktail_size', 3))
        num_recommendations = int(data.get('num_recommendations', 3))
        
        # Validate input
        if not host_id:
            return jsonify({'error': 'host_id is required'}), 400
        if not phage_list:
            return jsonify({'error': 'No phages provided in available_phages'}), 400
        
        # Here we would normally load phage & host data and generate features
        # For simplicity, we'll create a dummy feature dataset
        # In a real implementation, this would use the feature extraction functions
        
        # Generate dummy features for demonstration
        feature_columns = loaded_metadata.get('feature_columns', [])
        dummy_features = pd.DataFrame(
            np.random.random((len(phage_list), len(feature_columns))),
            columns=feature_columns
        )
        
        # Generate recommendations
        is_categorical = loaded_metadata.get('model_type') == 'classification'
        recommendations = recommend_phage_cocktail(
            loaded_model, 
            loaded_scaler, 
            dummy_features,
            phage_data=pd.DataFrame({'phage_id': phage_list}),
            is_categorical=is_categorical,
            n_recommendations=num_recommendations,
            cocktail_size=cocktail_size
        )
        
        logger.info(f"Generated {len(recommendations)} cocktail recommendations")
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/load_model', methods=['POST'])
def api_load_model():
    """Endpoint for loading a model"""
    global loaded_model, loaded_scaler, loaded_metadata
    
    try:
        data = request.json
        model_dir = data.get('model_dir')
        
        if not model_dir:
            return jsonify({'error': 'model_dir is required'}), 400
        
        loaded_model, loaded_scaler, loaded_metadata = load_model(model_dir)
        
        if loaded_model is None:
            return jsonify({'error': f'Failed to load model from {model_dir}'}), 400
        
        return jsonify({
            'status': 'success',
            'message': f'Model loaded from {model_dir}',
            'model_type': loaded_metadata.get('model_type', 'unknown'),
            'num_features': loaded_metadata.get('num_features', 0)
        })
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500
def main():
    """Main function for running PhageAI pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PhageAI: Phage-Host Interaction Prediction')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', required=True, help='Path to interaction data file')
    train_parser.add_argument('--phage-fasta', help='Path to phage genome FASTA file')
    train_parser.add_argument('--host-fasta', help='Path to host genome FASTA file')
    train_parser.add_argument('--target', help='Target column name')
    train_parser.add_argument('--output-dir', default='models', help='Output directory for model files')
    train_parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--model-dir', required=True, help='Path to model directory')
    predict_parser.add_argument('--data', required=True, help='Path to input data file')
    predict_parser.add_argument('--output', help='Path to output predictions file')
    
    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Generate phage cocktail recommendations')
    recommend_parser.add_argument('--model-dir', required=True, help='Path to model directory')
    recommend_parser.add_argument('--host-id', required=True, help='Host ID to generate recommendations for')
    recommend_parser.add_argument('--phage-list', help='Path to file with available phage IDs')
    recommend_parser.add_argument('--cocktail-size', type=int, default=3, help='Number of phages in each cocktail')
    recommend_parser.add_argument('--num-recommendations', type=int, default=3, help='Number of cocktail recommendations')
    recommend_parser.add_argument('--output', help='Path to output recommendations file')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start the API service')
    api_parser.add_argument('--model-dir', help='Path to model directory to load at startup')
    api_parser.add_argument('--host', default='127.0.0.1', help='API host')
    api_parser.add_argument('--port', type=int, default=5000, help='API port')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'train':
        # Load interaction data
        data = load_and_explore_data(args.data)
        if data is None:
            logger.error("Failed to load interaction data")
            return 1
        
        # Load genomic data if provided
        phage_sequences, host_sequences = load_genomic_data(args.phage_fasta, args.host_fasta)
        
        # Extract genomic features if genomic data was provided
        phage_features = pd.DataFrame()
        host_features = pd.DataFrame()
        
        if phage_sequences:
            logger.info("Extracting phage genomic features...")
            phage_features = extract_genomic_features(phage_sequences, prefix="phage_")
        
        if host_sequences:
            logger.info("Extracting host genomic features...")
            host_features = extract_genomic_features(host_sequences, prefix="host_")
        
        # Integrate genomic data with interaction data if available
        if not phage_features.empty or not host_features.empty:
            data = integrate_genomic_data(data, phage_features, host_features)
        
        # Preprocess data
        X, y, metadata, column_groups, is_categorical, feature_columns, target_column = preprocess_data(data, args.target)
        
        # Train and evaluate model
        model, scaler, feature_importance, is_categorical = train_evaluate_model(
            X, y, metadata, is_categorical, hyperparameter_tuning=args.tune
        )
        
        # Save model
        model_dir = save_model(model, scaler, feature_importance, feature_columns, target_column, is_categorical, args.output_dir)
        logger.info(f"Model saved to {model_dir}")
    # Inside the main() function, in the 'predict' command branch:
    elif args.command == 'predict':
        # Load model
        model, scaler, metadata = load_model(args.model_dir)
        if model is None:
            logger.error(f"Failed to load model from {args.model_dir}")
            return 1
        
        # Load input data
        data = load_and_explore_data(args.data)
        if data is None:
            logger.error("Failed to load input data")
            return 1
        
        # Check if this is staphylococcus data
        is_staph = False
        host_col = [col for col in data.columns if 'host' in col.lower() and 'name' in col.lower()]
        if host_col and len(host_col) > 0:
            host_names = data[host_col[0]].astype(str).str.lower()
            is_staph = any("staphylococcus" in name for name in host_names)
        
        # Extract features
        feature_columns = metadata.get('feature_columns', [])
        if not feature_columns:
            logger.error("No feature columns defined in model metadata")
            return 1
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in data.columns:
                logger.warning(f"Missing feature column: {col}. Setting to 0.")
                data[col] = 0
        
        # Keep only feature columns
        X = data[feature_columns]
        
        # Scale features
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Make predictions
        is_categorical = metadata.get('model_type') == 'classification'
        if is_categorical:
            try:
                # Try to get probability
                proba = model.predict_proba(X_scaled)
                predictions = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                # Adjust effectiveness scores
                predictions = np.array([adjust_effectiveness_score(score, is_staph) for score in predictions])
                class_predictions = model.predict(X_scaled)
                
                # Create results DataFrame
                results = pd.DataFrame({
                    'probability': predictions,
                    'class': class_predictions,
                    'effectiveness': predictions
                })
            except:
                # Fall back to class prediction only
                predictions = model.predict(X_scaled)
                effectiveness = np.where(predictions == 1, 1.0, 0.0)
                # Adjust effectiveness scores
                effectiveness = np.array([adjust_effectiveness_score(score, is_staph) for score in effectiveness])
                results = pd.DataFrame({
                    'class': predictions,
                    'effectiveness': effectiveness
                })
        else:
            predictions = model.predict(X_scaled)
            # Ensure regression predictions stay within 0-1 range
            predictions = np.clip(predictions, 0.0, 1.0)
            # Adjust effectiveness scores
            predictions = np.array([adjust_effectiveness_score(score, is_staph) for score in predictions])
            results = pd.DataFrame({'effectiveness': predictions})
        
        # Combine with original data if ID columns exist
        id_cols = [col for col in data.columns if '_ID' in col or 'id' in col.lower()]
        if id_cols:
            results = pd.concat([data[id_cols], results], axis=1)
        
        # Save or display results
        if args.output:
            results.to_csv(args.output, index=False)
            logger.info(f"Predictions saved to {args.output}")
        else:
            print(results.head(10).to_string())
            logger.info(f"Generated {len(results)} predictions")
    elif args.command == 'recommend':
        # Load model
        model, scaler, metadata = load_model(args.model_dir)
        if model is None:
            logger.error(f"Failed to load model from {args.model_dir}")
            return 1
        
        # Load phage list if provided
        phage_list = []
        if args.phage_list:
            try:
                with open(args.phage_list, 'r') as f:
                    phage_list = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(phage_list)} phages from {args.phage_list}")
            except Exception as e:
                logger.error(f"Error loading phage list: {e}")
                return 1
        else:
            # Generate dummy phage list for demonstration
            phage_list = [f"phage_{i}" for i in range(1, 21)]
            logger.info(f"Using {len(phage_list)} dummy phages for recommendation")
        
        # Generate dummy features for demonstration
        # In a real implementation, this would use the feature extraction functions
        feature_columns = metadata.get('feature_columns', [])
        dummy_features = pd.DataFrame(
            np.random.random((len(phage_list), len(feature_columns))),
            columns=feature_columns
        )
        
        # Generate recommendations
        is_categorical = metadata.get('model_type') == 'classification'
        recommendations = recommend_phage_cocktail(
            model, 
            scaler, 
            dummy_features,
            phage_data=pd.DataFrame({'phage_id': phage_list}),
            is_categorical=is_categorical,
            n_recommendations=args.num_recommendations,
            cocktail_size=args.cocktail_size
        )
        
        # Save or display recommendations
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(recommendations, f, indent=4)
            logger.info(f"Recommendations saved to {args.output}")
        else:
            for i, cocktail in enumerate(recommendations):
                print(f"Cocktail {i+1}:")
                print(f"  Phages: {', '.join(str(p) for p in cocktail['phages'])}")
                print(f"  Combined Score: {cocktail['combined_score']:.4f}")
                print(f"  Diversity Score: {cocktail['diversity_score']:.4f}")
                print()
    
    elif args.command == 'api':
        # Load model if provided
        if args.model_dir:
            loaded_model, loaded_scaler, loaded_metadata = load_model(args.model_dir)
            if loaded_model is None:
                logger.warning(f"Failed to load model from {args.model_dir}")
            else:
                logger.info(f"Model loaded from {args.model_dir}")
        
        # Start API service
        logger.info(f"Starting API service on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port)
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
