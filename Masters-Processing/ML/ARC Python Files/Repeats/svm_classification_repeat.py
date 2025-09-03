import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import optuna
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
import sys
import os

## IMPORT MRMR WRAPPER FROM PARENT DIRECTORY
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Add parent directory to Python path if not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from mrmr_wrapper import MRMRTransformer

if len(sys.argv)<1: 
    raise Exception 

# Load the data
file_path = sys.argv[1] # Path to the input CSV file
data = pd.read_csv(file_path)

# Shuffle the data
shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True) # frac=1 means 100% of data
data_shuffled = shuffled.iloc[:, 4:] # Drop the first 4 columns (unneeded metadata)
labels_shuffled = shuffled["Comfort Score"]

# Create binary labels (1,2 = 0; 4,5 = 1; exclude 3 for clearer separation)
binary_labels = labels_shuffled.apply(lambda x: 0 if x <= 2 else (1 if x >=4 else np.nan))
binary_data = data_shuffled[~binary_labels.isna()]
binary_labels = binary_labels[~binary_labels.isna()] # The "~" operator is used to filter out NaN values (i.e., rows where Comfort Score == 3)

# For binary classification
X_train, X_test, y_train, y_test = train_test_split(
    binary_data,
    binary_labels,
    test_size=0.2,
    stratify=binary_labels,
    random_state=42
)

# Handle missing values with median imputation
imputer = SimpleImputer(strategy='median')

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Scale the data (recommended for SVM)
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))

# Optimize Feature Selection and SVM Parameters    
X = X_train_scaled.copy()
y = y_train.copy()

def binary_classification_objective(trial):
    # Feature selection, only optimizing method and number of features (no hyperparameters of the methods)
    fs_method = trial.suggest_categorical('feature_selection', ['MRMR', 'RFE', 'None'])
    
    if fs_method != 'None':
        k_features = trial.suggest_int('k_features', 5, 105, step = 10) # Only allow a maximum of 105 features to be selected, with a step of 10, to go up to all features: X.shape[1]
        if fs_method == 'RFE':
            estimator = RandomForestClassifier()
            selector = RFE(estimator, n_features_to_select=k_features)
        else: #MRMR
            selector = MRMRTransformer(k_features=k_features) #https://feature-engine.trainindata.com/en/1.8.x/api_doc/selection/MRMR.html#feature_engine.selection.MRMR
    else:
        selector = 'passthrough'

    # SVM hyperparameters 
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']) # do this first so that degree can be conditionally added later     
    params = {
        'C': trial.suggest_float('C', 0.1, 100, log=True),
        'kernel': kernel,
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'degree': trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3,
        'probability': True,
        'random_state': 42
        }
    model = SVC(**params)
        
    # Pipeline
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('model', model)
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        return np.mean(scores)
    except Exception as e:
        print(f"Error in trial: {e}")
        return -np.inf
    
# Create a list to store results from all runs
all_results = []

# Repeat the optimization and testing process 3 times
for run in range(3):
    print(f"\n{'='*50}")
    print(f"STARTING RUN {run + 1}/3")
    print(f"{'='*50}")
    
    # Run binary classification study
    study = optuna.create_study(direction='maximize')
    study.optimize(binary_classification_objective, n_trials=200, show_progress_bar=True)

    # Print results
    print("\nBinary Classification Optimization Results:")
    print(f"Best ROC AUC Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # apply feature selection code from before
    best_fs_method = study.best_params.get('feature_selection', 'None')

    if best_fs_method != 'None':
        k_features = study.best_params['k_features']
        if best_fs_method == 'RFE':
            estimator = RandomForestClassifier()
            selector = RFE(estimator, n_features_to_select=k_features)
        elif best_fs_method == 'MRMR':
            selector = MRMRTransformer(k_features=k_features)
        
        selector.fit(X, y)
        if hasattr(selector, 'get_support'):  # For RFE
            selected_features = X.columns[selector.get_support()]
        else:  # For MRMR
            selected_features = selector.selected_features
        X_best = X[selected_features]
    else:
        X_best = X
        selected_features = X.columns

    # Apply the same feature selection to test data
    if best_fs_method != 'None':
        if best_fs_method == 'MRMR':
            X_test_final = X_test[selected_features]
        else:
            X_test_final = selector.transform(X_test)
            if isinstance(X_test, pd.DataFrame):
                X_test_final = pd.DataFrame(X_test_final, columns=selected_features)
    else:
        X_test_final = X_test

    # Create and Fit SVM Model with Best Parameters
    best_model = SVC(
        C=study.best_params['C'],
        kernel=study.best_params['kernel'],
        gamma=study.best_params['gamma'],
        degree=study.best_params.get('degree', 3),  # Default degree if not in params
        probability=True,
        random_state=42
    )

    # Train on full imputed data
    best_model.fit(X_best, y)

    # Make predictions
    y_pred = best_model.predict(X_test_final)
    y_pred_proba = best_model.predict_proba(X_test_final)[:, 1]  # Probabilities for class 1

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Store results for this run
    run_result = {
        'run_number': run + 1,
        'best_cv_score': study.best_value,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    # Add all parameters to the result
    for key, value in study.best_params.items():
        run_result[f'param_{key}'] = value
    
    all_results.append(run_result)

    print("\nFinal Model Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

# Calculate averages across all runs
print(f"\n{'='*50}")
print("SUMMARY ACROSS ALL RUNS")
print(f"{'='*50}")

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(all_results)

# Calculate averages for metrics
metrics_to_avg = ['best_cv_score', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Create a summary row
summary_row = {'run_number': 'SUMMARY'}
for metric in metrics_to_avg:
    avg_value = results_df[metric].mean()
    std_value = results_df[metric].std()
    summary_row[metric] = f"{avg_value:.4f} ± {std_value:.4f}"
    print(f"{metric}: {avg_value:.4f} ± {std_value:.4f}")

# For parameters, just mark as N/A in summary
for key in study.best_params.keys():
    summary_row[f'param_{key}'] = 'N/A'

# Combine individual results with summary
combined_results = all_results + [summary_row]

# Save to CSV
df = pd.DataFrame(combined_results)
df.to_csv(sys.argv[2], index=False)

print(f"\nCombined results with averages saved to: {sys.argv[2]}")