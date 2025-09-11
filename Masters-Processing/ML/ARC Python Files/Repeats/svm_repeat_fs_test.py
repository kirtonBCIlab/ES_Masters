import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import optuna
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.inspection import permutation_importance

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

# Scale (keep column names)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Use these for optimization
X = X_train_scaled.copy()
y = y_train.copy()

def binary_classification_objective(trial):
    # Feature selection method
    fs_method = trial.suggest_categorical('feature_selection', ['MRMR', 'RFE', 'None'])

    # Robust k_features bounds
    n_total_features = X.shape[1]
    max_k_allowed = min(105, n_total_features)
    min_k_allowed = 1 if n_total_features < 5 else 5
    if fs_method != 'None' and max_k_allowed >= min_k_allowed:
        # step 10 only if range large enough; otherwise just search full int range
        if max_k_allowed - min_k_allowed >= 10:
            k_features = trial.suggest_int('k_features', min_k_allowed, max_k_allowed, step=10)
        else:
            k_features = trial.suggest_int('k_features', min_k_allowed, max_k_allowed)
        if fs_method == 'RFE':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=k_features)
        else:
            selector = MRMRTransformer(k_features=k_features)
    else:
        selector = 'passthrough'

    # SVM hyperparameters
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    params = {
        'C': trial.suggest_float('C', 0.1, 100, log=True),
        'kernel': kernel,
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
        'degree': trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3,
        'probability': True,
        'random_state': 42
    }
    model = SVC(**params)

    pipeline = Pipeline([
        ('feature_selection', selector),
        ('model', model)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        return np.mean(scores)
    except Exception as e:
        # Print the error for debugging and return a very poor score
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
    #selected_features = None

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
            X_test_final = X_test_scaled[selected_features]  # This should be DataFrame
        else:  # RFE
            X_test_transformed = selector.transform(X_test_scaled)
            # Convert back to DataFrame with feature names
            X_test_final = pd.DataFrame(X_test_transformed, columns=selected_features)
    else:
        X_test_final = X_test_scaled  # DataFrame

    # ---------------------------
    # Train final SVM with best params and evaluate on test set
    # ---------------------------
    best_model_params = {
        'C': study.best_params['C'],
        'kernel': study.best_params['kernel'],
        'gamma': study.best_params['gamma'],
        'probability': True,
        'random_state': 42
    }
    if best_model_params['kernel'] == 'poly':
        best_model_params['degree'] = study.best_params.get('degree', 3)

    best_model = SVC(**best_model_params)
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
    # FEATURE IMPORTANCE for this run
    # ---------------------------
    print("\n" + "="*30)
    print(f"FEATURE IMPORTANCE (run {run+1})")
    print("="*30)

    # Always convert selected_features → indices relative to binary_data.columns
    if best_fs_method != 'None':
        if isinstance(selected_features[0], str):
            # gave names → get indices
            selected_indices = [binary_data.columns.get_loc(feat) for feat in selected_features]
        else:
            # Already indices
            selected_indices = list(selected_features)
    else:
        selected_indices = list(range(binary_data.shape[1]))

    # Convert back indices → names
    final_feature_names = binary_data.columns[selected_indices].tolist()

    n_features_in_model = len(final_feature_names)
    print(f"Number of features in final model: {n_features_in_model}")

    if best_model.kernel == 'linear':
        # Use coefficients
        try:
            coef = best_model.coef_.ravel()
        except Exception as e:
            print(f"Couldn't extract coef_ for linear kernel: {e}")
            coef = np.zeros(n_features_in_model)

        feature_importance_df = pd.DataFrame({
            'feature': final_feature_names,
            'index': selected_indices,
            'coefficient': coef
        })
        feature_importance_df['abs_importance'] = feature_importance_df['coefficient'].abs()
        feature_importance_df = feature_importance_df.sort_values('abs_importance', ascending=False).reset_index(drop=True)

        # top-20
        top_n = min(20, feature_importance_df.shape[0])
        top_features = feature_importance_df.head(top_n).copy()
        top_features['run'] = run + 1
        top_features['best_cv_score'] = study.best_value

        # Save per-run CSV and PNG
        run_csv = os.path.join(out_dir, f'feature_importance_run{run+1}.csv')
        feature_importance_df.to_csv(run_csv, index=False)
        print(f"Saved feature importance CSV for run {run+1} -> {run_csv}")

        # Optionally save a plot (PNG)
        try:
            plt.figure(figsize=(8, 6))
            colors = ['red' if x < 0 else 'blue' for x in feature_importance_df['coefficient'].head(top_n)]
            plt.barh(range(top_n), feature_importance_df['coefficient'].head(top_n), color=colors)
            plt.yticks(range(top_n), feature_importance_df['feature'].head(top_n))
            plt.gca().invert_yaxis()
            plt.xlabel('Coefficient Value')
            plt.title(f'Top {top_n} Feature Coefficients (run {run+1})')
            plt.tight_layout()
            png_path = os.path.join(out_dir, f'feature_importance_run{run+1}.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Couldn't save plot for run {run+1}: {e}")

    else:
        # Non-linear -> permutation importance
        try:
            perm = permutation_importance(
                best_model,
                X_test_final,
                y_test,
                n_repeats=50,
                random_state=42,
                scoring='roc_auc',
                n_jobs=4
            )
            importance_df = pd.DataFrame({
                'feature': final_feature_names,
                'index': selected_indices,
                'importance_mean': perm.importances_mean,
                'importance_std': perm.importances_std
            })
            importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
           
            top_n = min(20, importance_df.shape[0])
            top_features = importance_df.head(top_n).copy()
            top_features['run'] = run + 1
            top_features['best_cv_score'] = study.best_value

            run_csv = os.path.join(out_dir, f'feature_importance_run{run+1}.csv')
            importance_df.to_csv(run_csv, index=False)
            print(f"Saved permutation importance CSV for run {run+1} -> {run_csv}")

            # Save plot
            try:
                plt.figure(figsize=(8, 6))
                plt.barh(range(top_n), importance_df['importance_mean'].head(top_n),
                         xerr=importance_df['importance_std'].head(top_n), capsize=3)
                plt.yticks(range(top_n), importance_df['feature'].head(top_n))
                plt.gca().invert_yaxis()
                plt.xlabel('Permutation Importance (Mean decrease in ROC AUC)')
                plt.title(f'Top {top_n} Feature Importances (run {run+1})')
                plt.tight_layout()
                png_path = os.path.join(out_dir, f'feature_importance_run{run+1}.png')
                plt.savefig(png_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Couldn't save permutation plot for run {run+1}: {e}")

        except Exception as e:
            print(f"Permutation importance failed for run {run+1}: {e}")
            # Create a fallback empty frame
            importance_df = pd.DataFrame({'feature': final_feature_names, 'importance_mean': 0.0})
            top_features = importance_df.head(min(20, len(final_feature_names))).copy()
            top_features['run'] = run + 1
            top_features['best_cv_score'] = study.best_value
            run_csv = os.path.join(out_dir, f'feature_importance_run{run+1}.csv')
            importance_df.to_csv(run_csv, index=False)

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
