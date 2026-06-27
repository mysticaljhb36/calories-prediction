# =============================================================================
# Packages Setup
# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)

import json
import shap
from datetime import datetime
import logging
import warnings
from sklearn.exceptions import DataConversionWarning

from feature_engineering import feature_creation
from paths import RAW_DIR, MODELS_DIR, REPORTS_DIR
from utils import data_path, transform_imputer
# Import and run reusable logger module to create pipeline.log.
from logger import loggers

# Create module-level logger for training-specific pipeline events.
logger = logging.getLogger(__name__)
# Suppress sklearn warnings caused by passing a single-column dataframe
# instead of a 1D target array. This keeps logs cleaner during training.
warnings.filterwarnings(
    "ignore",
    category=DataConversionWarning
)

# --------------------------------------------------
# Timestamp
# --------------------------------------------------
    
timestamp = datetime.now().strftime("%Y%m%d %H%M%S")

# --------------------------------------------------
# Load impulse data from raw directory 
# --------------------------------------------------
@transform_imputer
def load_data(filename: str, subdir) -> pd.DataFrame:
    """
    Load raw training data and apply imputation using the project imputer.

    Args:
        filename (str):
            Name of the raw dataset.

        subdir:
            Directory where the dataset is stored.

    Returns:
        pd.DataFrame:
            Raw dataset after missing-value imputation.
    """

    logger.info("Loading raw training dataset.")

    df = pd.read_csv(
        data_path(filename, subdir)
    )

    if df.empty:
        logger.error("Training dataset is empty.")
        raise ValueError("Training dataset is empty.")

    logger.info(
        f"Training dataset loaded successfully. Shape: {df.shape}"
    )

    return df

# --------------------------------------------------
# Prepare data for training
# --------------------------------------------------
def prepare_training_data(df: pd.DataFrame):
    """
    Create model features and split data into train and test sets.

    Args:
        df (pd.DataFrame):
            Raw training dataframe.

    Returns:
        tuple:
            X_train, X_test, y_train, y_test
    """

    logger.info("Starting feature creation for training data.")

    features_df = feature_creation(
        df.copy()
    )

    if "Calories" not in features_df.columns:
        logger.error("Target column 'Calories' is missing.")
        raise KeyError("Target column 'Calories' is missing.")

    # Separate independent variables from the target variable.
    # This prevents the model from learning from the answer column.
    X = features_df.drop(
        columns="Calories"
    )

    y = features_df["Calories"]

    logger.info(
        f"Feature matrix shape: {X.shape}. "
        f"Target shape: {y.shape}."
    )

    # Keep a fixed random_state so model evaluation is reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    logger.info(
        f"Train/test split complete. "
        f"X_train: {X_train.shape}, X_test: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test

# --------------------------------------------------
# Power Transform Features 
# --------------------------------------------------
def transform_features(X_train, X_test):
    """
    Fit the transformer on training data and apply it to test data.

    Args:
        X_train:
            Training features.

        X_test:
            Test features.

    Returns:
        tuple:
            X_train_scaled, X_test_scaled, fitted PowerTransformer
    """

    logger.info("Applying PowerTransformer to training and test data.")

    ptransformer = PowerTransformer()

    # Golden rule:
    # fit_transform() is used only on training data because the transformer
    # must learn scaling parameters from training data only.
    X_train_scaled = ptransformer.fit_transform(
        X_train
    )

    # transform() is used on test data to avoid data leakage.
    X_test_scaled = ptransformer.transform(
        X_test
    )

    logger.info("Feature transformation completed successfully.")

    return X_train_scaled, X_test_scaled, ptransformer

# --------------------------------------------------
# Hyperparameter Tuning 
# --------------------------------------------------
def tune_random_forest(X_train_scaled, y_train):
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.

    Args:
        X_train_scaled:
            Transformed training features.

        y_train:
            Training target values.

    Returns:
        RandomizedSearchCV:
            Fitted hyperparameter search object.
    """

    logger.info("Starting Random Forest hyperparameter tuning.")

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [10, 15, 20, 25],
        "max_depth": [None, 3, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=50,
        scoring="r2",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    rf_search.fit(
        X_train_scaled,
        y_train
    )

    logger.info(
        f"Hyperparameter tuning completed. "
        f"Best CV R2: {rf_search.best_score_:.4f}"
    )

    logger.info(
        f"Best parameters: {rf_search.best_params_}"
    )

    return rf_search

# --------------------------------------------------
# Evaluate Model 
# --------------------------------------------------
def evaluate_model(y_test, predictions) -> dict:
    """
    Evaluate model predictions using regression metrics.

    Args:
        y_test:
            True target values.

        predictions:
            Predicted target values.

    Returns:
        dict:
            Dictionary containing R2, MAE and RMSE.
    """

    return {
        "R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": root_mean_squared_error(y_test, predictions)
    }

# ---------------------------------------------------------------------------
# Train Model
# ---------------------------------------------------------------------------
def train_model(raw_data):
    """
    Run the full model training pipeline.

    This function loads data, creates features, transforms features,
    tunes a Random Forest model, evaluates performance and saves the
    trained model and fitted transformer.

    Returns:
        dict:
            Model evaluation results.
    """

    logger.info("Model training pipeline started.")

    try:

        X_train, X_test, y_train, y_test = prepare_training_data(
            raw_data
        )

        X_train_scaled, X_test_scaled, ptransformer = transform_features(
            X_train,
            X_test
        )

        rf_search = tune_random_forest(
            X_train_scaled,
            y_train
        )

        best_rf = rf_search.best_estimator_

        logger.info("Generating predictions using tuned Random Forest.")

        tuned_predictions = best_rf.predict(
            X_test_scaled
        )

        results = evaluate_model(
            y_test,
            tuned_predictions
        )

        logger.info(
            f"Tuned Random Forest results: {results}"
        )

        # Save the fitted transformer used during training.
        # The prediction pipeline must reuse this exact transformer
        # to keep training and inference transformations consistent.
        joblib.dump(
            ptransformer,
            MODELS_DIR / "power_transformer.pkl"
        )

        logger.info("PowerTransformer saved successfully.")

        # Save the trained model for later inference.
        joblib.dump(
            best_rf,
            MODELS_DIR / "random_forest_model.pkl"
        )

        logger.info("Random Forest model saved successfully.")

        logger.info("Model training pipeline completed successfully.")

        return results, best_rf

    except Exception as error:
        logger.error(
            f"Model training pipeline failed: {error}"
        )
        raise


# --------------------------------------------------
# Save shap_importance_.csv
# --------------------------------------------------
def save_shap_importance(
        model,
        X_train,
        reports_dir,
        datetime_obj) -> None:
    """
    Save mean absolute SHAP importance.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_train)

    shap_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "mean_abs_shap":
            abs(shap_values).mean(axis=0)
    }).sort_values(
        by="mean_abs_shap",
        ascending=False
    )

    shap_importance_df.round(4).to_csv(
        reports_dir / f"shap_importance_{datetime_obj}.csv",
        index=False
    )
    


# --------------------------------------------------
# Save feature_importance.csv
# --------------------------------------------------    
def save_feature_importance(
        model,
        X_train,
        reports_dir,
        datetime_obj) -> None:
    """
    Save feature importance.
    """

    feature_importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }
        ).sort_values(
            by="importance",
            ascending=False
    )
    
    feature_importance_df.round(4).to_csv(
        reports_dir / f"feature_importance_{datetime_obj}.csv",
        index=False
    )
    
   
# --------------------------------------------------
# Save model_metrics_.csv
# --------------------------------------------------
def save_model_metrics(
        datetime_obj,
        X_train,
        X_test,
        reports_dir,
        **metrics: dict
        ) -> None:
    """
    SUMMARY: Save model importance.

    Returns
    -------
    None
    """
    metrics_df = pd.DataFrame(
        {
            "model": ["RandomForestRegressor"],
            "r2_score": [metrics['R2']],
            "mae": [metrics['MAE']],
            "rmse": [metrics['RMSE']],
            "train_rows": [len(X_train)],
            "test_rows": [len(X_test)],
            "run_timestamp": [datetime_obj]
        }
    )
    
    metrics_df.round(4).to_csv(
        reports_dir / f"model_metrics_{datetime_obj}.csv",
        index=False
    )
    

# --------------------------------------------------
# Save training_summary.json
# --------------------------------------------------
def save_training_summary(
        model: object,
        X_train,
        X_test,        
        reports_dir: str,
        datetime_obj: object,
        **metrics: dict
        ) -> None:
    training_summary = {
        "model_name": "RandomForestRegressor",
        "training_timestamp": datetime_obj,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "number_of_features": len(X_train.columns),
        "features": list(X_train.columns),
        "metrics": {
            "r2_score": float(metrics['R2']),
            "mae": float(metrics['MAE']),
            "rmse": float(metrics['RMSE'])
        },
        "hyperparameters": model.get_params()
    }
    
    with open(
        reports_dir / f"figures/training_summary_{timestamp}.json",
        "w"
    ) as f:
    
        json.dump(
            training_summary,
            f,
            indent=4,
            default=str
        )



if __name__ == "__main__":
    
    impulse_data = load_data(
        "impulse_data",
        RAW_DIR
    )

    output, best_model = train_model(
        impulse_data
    )
    
    
    X_train, X_test, y_train, y_test = prepare_training_data(
        impulse_data
    )    
  
    save_shap_importance(
        best_model, 
        X_train, 
        REPORTS_DIR,
        timestamp
    )
    
    save_feature_importance(
            best_model,
            X_train,
            REPORTS_DIR,
            timestamp
    )
    
    save_model_metrics(
            timestamp,
            X_train,
            X_test,
            REPORTS_DIR,
            **output
    )
    
    save_training_summary(
        best_model,
        X_train,
        X_test,        
        REPORTS_DIR,
        timestamp,
        **output
    )