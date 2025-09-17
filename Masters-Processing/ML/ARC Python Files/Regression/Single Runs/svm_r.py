import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import optuna
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from mrmr_wrapper import MRMRTransformer
import sys

if len(sys.argv)<1:
    raise Exception

# Load the data
file_path = sys.argv[1]
data = pd.read_csv(file_path)

# Shuffle the data
shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
data_shuffled = shuffled.iloc[:, 4:]
labels_shuffled = shuffled["Comfort Score"]

# For regression (using stratified split based on binned target)
X_train, X_test, y_train, y_test = train_test_split(
    data_shuffled, 
    labels_shuffled, 
    test_size=0.2, 
    stratify=labels_shuffled,
    random_state=42
)

imputer = SimpleImputer(strategy='median')

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
X_test_scaled = pd.DataFrame(scaler.transform(X_test))

X = X_train_scaled.copy()
y = y_train.copy()

def objective(trial):
    # Feature selection, only optimizing method and number of features (no hyperparameters of the methods)
    fs_method = trial.suggest_categorical('feature_selection', ['RFE', 'MRMR', 'None'])

    if fs_method != 'None':
        k_features = trial.suggest_int('k_features', 5, 105, step = 10) 
        if fs_method == 'RFE':
            estimator = RandomForestRegressor()
            selector = RFE(estimator, n_features_to_select=k_features)
        else: #MRMR
            selector = MRMRTransformer(k_features=k_features) #https://feature-engine.trainindata.com/en/1.8.x/api_doc/selection/MRMR.html#feature_engine.selection.MRMR
    else:
        selector = 'passthrough'

    # SVM Regression Hyperparameters
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    
    if kernel in ['rbf', 'poly']:
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.01, 0.1, 1])
    else:
        gamma = 'scale'  # default for linear kernel

    # Degree only applies for polynomial kernel
    degree = trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3

    # Hyperparameters dictionary
    params = {
        'C': trial.suggest_float('C', 0.1, 100, log=True),
        'kernel': kernel,
        'gamma': gamma,
        'degree': degree,
        'epsilon': trial.suggest_float('epsilon', 0.01, 0.5)
    }
    model = SVR(**params)

    # 3. Pipeline 
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('model', model)
    ])

    #4. Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2', n_jobs=1)
        return np.mean(scores)
    except Exception:
        return -np.inf

# Run Optuna Study
study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=200, show_progress_bar=True, n_jobs=5) 

# Best result
print("\n Regression Optimization Results")
print("===================================")
print(f"Best R²: {study.best_value:.4f}")
print("Best Parameters: ")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# apply your feature selection code from before
best_fs_method = study.best_params.get('feature_selection', 'None')

if best_fs_method != 'None':
    k_features = study.best_params['k_features']
    if best_fs_method == 'RFE':
        estimator = RandomForestRegressor()
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

best_model = SVR(
    C=study.best_params['C'],
    kernel=study.best_params['kernel'],
    gamma=study.best_params['gamma'],
    degree=study.best_params.get('degree', 3),  # Default degree if not in params
)

# Train on full (potentially feature-selected) data
best_model.fit(X_best, y)
params_dict = best_model.get_params()

# Make predictions
y_pred = best_model.predict(X_test_final)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nFinal Model Evaluation on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Calculate accuracy within ±1 point
correct = np.sum(np.abs(y_test - y_pred) <= 1)
accuracy = correct / len(y_test)
print(f"Accuracy within ±1 point: {accuracy:.4f}")