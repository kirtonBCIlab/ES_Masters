import numpy as np
import pandas as pd
import optuna
from mrmr_wrapper import MRMRTransformer
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data_shuffled, 
    labels_shuffled, 
    test_size=0.2, 
    stratify=labels_shuffled,
    random_state=42
)

# Optimization Feature Selection + CatBoost Regressor
X = X_train.copy()
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

    # CatBoost Hyperparameters
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=25),
        'depth': trial.suggest_int('depth', 6, 10), #https://catboost.ai/docs/en/concepts/parameter-tuning
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True), #https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True), 
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'verbose': False,
    }
    model = CatBoostRegressor(**params)

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
study.optimize(objective, n_trials=200, show_progress_bar=True, n_jobs=6) 

# Best result
print("\n Regression Optimization Results")
print("===================================")
print(f"Best R²: {study.best_value:.4f}")
print("Best Parameters: ")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Apply best feature selection to training and test data
best_fs_method = study.best_params.get('feature_selection', 'None')

if best_fs_method != 'None':
    k_features = study.best_params['k_features']
    if best_fs_method == 'RFE':
        estimator = RandomForestRegressor()
        selector = RFE(estimator, n_features_to_select=k_features,)
    else: #MRMR
        selector = MRMRTransformer(k_features=k_features)
    
    selector.fit(X, y)
    if hasattr(selector, 'get_support'):  # For SelectKBest/RFE
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
        X_test_final = selector.transform(X_test)  # Use the already fitted selector
        if isinstance(X_test, pd.DataFrame):
            X_test_final = pd.DataFrame(X_test_final, columns=selected_features)
else:
    X_test_final = X_test

# Create and train the final model with the best parameters
best_model = CatBoostRegressor(
    iterations=study.best_params['iterations'],
    depth=study.best_params['depth'],
    learning_rate=study.best_params['learning_rate'],
    l2_leaf_reg= study.best_params['l2_leaf_reg'],
    random_strength=study.best_params['random_strength'],
    bagging_temperature=study.best_params['bagging_temperature'],
    border_count=study.best_params['border_count'],
    random_seed = 42,
    verbose=False
)

# Train on full imputed data
best_model.fit(X_best, y)

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