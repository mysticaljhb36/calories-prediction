# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:51:16 2026

@author: dooke
"""

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import PowerTransformer
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error
)

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)

from utils import data_path, transform_imputer
from paths import FEATURES_DIR, RAW_DIR, MODELS_DIR
from feature_engineering import feature_creation

@transform_imputer
def load_data(filename:str, subdir) -> pd.DataFrame:
    return pd.read_csv(data_path(filename, subdir))

X_raw = load_data("impulse_data", RAW_DIR)

X_features = feature_creation(X_raw.copy())

#Train Test Split
X, Y = X_features.drop(columns='Calories'), X_features.loc[:, ['Calories']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
  
#Data transformation
ptransformer = PowerTransformer()  

# Do not fit_transform on test data because the scaler already learnt from the training data
# Golden rule - fit() = learn, transform() - apply
X_train_scaled = ptransformer.fit_transform(X_train)
X_test_scaled = ptransformer.transform(X_test)    
# Save transformer 
#joblib.dump(ptransformer, MODELS_DIR / "power_transformer.pkl")


# Baseline model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

#######################################################
### Random Forest Hyper Porameter Tuning
# Hyperparameter search space
param_grid = {
    'n_estimators': [10, 15, 20, 30, 60],
    'max_depth': [None, 3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Randomized search
rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=50,
    scoring='r2',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit hyper tuned search
rf_search.fit(X_train_scaled, y_train)

# Best model
best_rf = rf_search.best_estimator_

print("Best Parameters:")
print(rf_search.best_params_)

print("Best CV R²:")
print(rf_search.best_score_)

#############################################################

# Fit baseline model
rf_model.fit(X_train_scaled, y_train)

# Test baseline model
rf_baseline = rf_model.predict(X_test_scaled)


# Optimised model
rf_optimised = RandomForestRegressor(
    n_estimators = 15, 
    min_samples_split = 3, 
    min_samples_leaf = 1, 
    max_features = 'sqrt', 
    max_depth = 5, 
    random_state=42,
    n_jobs=-1,
    bootstrap= False)
rf_optimised.fit(X_train_scaled, y_train)

# Test Optimised model
rf_tuned = rf_optimised.predict(X_test_scaled)

print(f"Baseline Model R²     : {r2_score(y_test, rf_baseline):.4f}")
print(f"Optimised Model R²     : {r2_score(y_test, rf_tuned):.4f}")

results = {
    "Baseline RF": {
        "R2": r2_score(y_test, rf_baseline),
        "MAE": mean_absolute_error(y_test, rf_baseline),
        "RMSE": root_mean_squared_error(y_test, rf_baseline)
    },
    "Tuned RF": {
        "R2": r2_score(y_test, rf_tuned),
        "MAE": mean_absolute_error(y_test, rf_tuned),
        "RMSE": root_mean_squared_error(y_test, rf_tuned)
    }
}

print(results)
# Save model
#joblib.dump(rf_optimised, MODELS_DIR / "random_forest_model.pkl")