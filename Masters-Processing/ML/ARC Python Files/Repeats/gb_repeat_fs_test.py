import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import optuna
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

## IMPORT MRMR WRAPPER FROM PARENT DIRECTORY
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from mrmr_wrapper import MRMRTransformer

# ---------------------------
# Args check
# ---------------------------
if len(sys.argv) < 3:
    print("Usage: python script.py <input_csv> <results_output_csv>")
    sys.exit(1)

file_path = sys.argv[1]
results_outpath = sys.argv[2]

# Directory to save per-run feature importance CSVs
out_dir = os.path.splitext(results_outpath)[0] + "_feature_importances"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
data = pd.read_csv(file_path)

# Shuffle the data
shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
data_shuffled = shuffled.iloc[:, 4:]  # drop the first 4 metadata columns as before
labels_shuffled = shuffled["Comfort Score"]

# Create binary labels (1,2 -> 0 ; 4,5 -> 1 ; drop 3)
binary_labels = labels_shuffled.apply(lambda x: 0 if x <= 2 else (1 if x >=4 else np.nan))
binary_data = data_shuffled[~binary_labels.isna()].reset_index(drop=True)
binary_labels = binary_labels[~binary_labels.isna()].reset_index(drop=True).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    binary_data,
    binary_labels,
    test_size=0.2,
    stratify=binary_labels,
    random_state=42
)

# Impute missing values (median)
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=binary_data.columns, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=binary_data.columns, index=X_test.index)


# Optimize Feature Selection and Gradient Boosting Parameters
X = X_train.copy()
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
    
    # Gradient Boosting hyperparameters https://www.geeksforgeeks.org/machine-learning/how-to-tune-hyperparameters-in-gradient-boosting-algorithm/, https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': 42
    }
    model = GradientBoostingClassifier(**params)
        
    # Pipeline
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('model', model)
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=4)
        return np.mean(scores)
    except Exception as e:
        print(f"Error in trial: {e}")
        return -np.inf
    
# Storage
all_results = []
top_features_all_runs = []

# Repeat optimization + evaluation 10 times
for run in range(10):
    print(f"\n{'='*50}")
    print(f"STARTING RUN {run + 1}/10")
    print(f"{'='*50}")

    study = optuna.create_study(direction='maximize')
    study.optimize(binary_classification_objective, n_trials=100, show_progress_bar=True, n_jobs=4)

    print("\nBinary Classification Optimization Results:")
    print(f"Best ROC AUC Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # ---------------------------
    # Recreate selector using best params and extract selected feature names
    # ---------------------------
    best_fs_method = study.best_params.get('feature_selection', 'None')
    selected_features = None

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
            X_test_final = X_test[selected_features]  # This should be DataFrame
        else:  # RFE
            X_test_transformed = selector.transform(X_test)
            # Convert back to DataFrame with feature names
            X_test_final = pd.DataFrame(X_test_transformed, columns=selected_features)
    else:
        X_test_final = X_test  # DataFrame

    # ---------------------------
    # Train final GB with best params and evaluate on test set
    # ---------------------------
    best_model_params = {
        'n_estimators' : study.best_params['n_estimators'],
        'max_depth' : study.best_params['max_depth'],
        'learning_rate' : study.best_params['learning_rate'],
        'min_samples_split' : study.best_params['min_samples_split'],
        'min_samples_leaf' : study.best_params['min_samples_leaf'],
        'subsample' : study.best_params['subsample'],
        'random_state' : 42
    } 
    
    best_model = GradientBoostingClassifier(**best_model_params)
    best_model.fit(X_best, y)

    # Predictions
    y_pred = best_model.predict(X_test_final)
    y_pred_proba = best_model.predict_proba(X_test_final)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Save run result
    run_result = {
        'run_number': run + 1,
        'best_cv_score': study.best_value,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'selected_features': ";".join(selected_features),
        'feature_selection_method': best_fs_method
    }
    for key, value in study.best_params.items():
        run_result[f'param_{key}'] = value

    all_results.append(run_result)

    print("\nFinal Model Evaluation on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # ---------------------------
    # FEATURE IMPORTANCE for Gradient Boosting
    # ---------------------------
    print("\n" + "="*30)
    print(f"FEATURE IMPORTANCE (run {run+1})")
    print("="*30)

    # Always convert selected_features → indices relative to binary_data.columns
    if best_fs_method != 'None':
        if isinstance(selected_features[0], str):
            selected_indices = [binary_data.columns.get_loc(feat) for feat in selected_features]
        else:
            selected_indices = list(selected_features)
    else:
        selected_indices = list(range(binary_data.shape[1]))

    # Convert back indices → names
    final_feature_names = binary_data.columns[selected_indices].tolist()

    n_features_in_model = len(final_feature_names)
    print(f"Number of features in final model: {n_features_in_model}")

    try:
        # Gradient Boosting has built-in feature importances
        importances = best_model.feature_importances_

        feature_importance_df = pd.DataFrame({
            'feature': final_feature_names,
            'index': selected_indices,
            'importance': importances
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        # Top-20
        top_n = min(20, feature_importance_df.shape[0])
        top_features = feature_importance_df.head(top_n).copy()
        top_features['run'] = run + 1
        top_features['best_cv_score'] = study.best_value

        # Save per-run CSV
        run_csv = os.path(f'feature_importance_run{run+1}.csv')
        feature_importance_df.to_csv(run_csv, index=False)
        print(f"Saved feature importance CSV for run {run+1} -> {run_csv}")

        # Save plot
        try:
            plt.figure(figsize=(8, 6))
            plt.barh(range(top_n), feature_importance_df['importance'].head(top_n))
            plt.yticks(range(top_n), feature_importance_df['feature'].head(top_n))
            plt.gca().invert_yaxis()
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Features (run {run+1})')
            plt.tight_layout()
            png_path = os.path.join(out_dir, f'feature_importance_run{run+1}.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Couldn't save plot for run {run+1}: {e}")

    except Exception as e:
        print(f"Failed to extract feature_importances_ for run {run+1}: {e}")
        feature_importance_df = pd.DataFrame({
            'feature': final_feature_names,
            'index': selected_indices,
            'importance': np.zeros(len(final_feature_names))
        })
        top_features = feature_importance_df.head(min(20, len(final_feature_names))).copy()
        top_features['run'] = run + 1
        top_features['best_cv_score'] = study.best_value
        run_csv = os.path(f'feature_importance_run{run+1}.csv')
        feature_importance_df.to_csv(run_csv, index=False)

    # Collect top features for final combined CSV
    top_features_all_runs.append(top_features)

# ---------------------------
# After all runs: save combined results & top features
# ---------------------------
print(f"\n{'='*50}")
print("SUMMARY ACROSS ALL RUNS")
print(f"{'='*50}")

results_df = pd.DataFrame(all_results)
metrics_to_avg = ['best_cv_score', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']

summary_row = {'run_number': 'SUMMARY'}
for metric in metrics_to_avg:
    avg_value = results_df[metric].mean()
    std_value = results_df[metric].std()
    summary_row[metric] = f"{avg_value:.4f} ± {std_value:.4f}"
    print(f"{metric}: {avg_value:.4f} ± {std_value:.4f}")

# Mark params N/A for summary
for key in [k for k in results_df.columns if k.startswith('param_')]:
    summary_row[key] = 'N/A'

combined_results = all_results + [summary_row]
df = pd.DataFrame(combined_results)
df.to_csv(results_outpath, index=False)
print(f"\nCombined results with averages saved to: {results_outpath}")

# Combine top features from all runs into one CSV
if top_features_all_runs:
    combined_top = pd.concat(top_features_all_runs, ignore_index=True, sort=False)
    combined_top_csv = os.path.join(out_dir, 'combined_top_features.csv')
    combined_top.to_csv(combined_top_csv, index=False)
    print(f"Combined top features saved to: {combined_top_csv}")
else:
    print("No top features were collected.")

print("Done.")
