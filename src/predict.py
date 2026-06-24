# =============================================================================
# Packages Setup
# =============================================================================

import logging
import pandas as pd

from paths import FEATURES_DIR, POWER_TRANSFORMER, RF_MODEL
from utils import data_path, load_model


# Create module-level logger.
# This identifies prediction-related messages in the pipeline log.
logger = logging.getLogger(__name__)


def read_feature_data(filename: str, subdir) -> pd.DataFrame:
    """
    Load engineered feature data for model inference.

    Args:
        filename (str):
            Name of the feature dataset to load.

        subdir:
            Directory where the feature dataset is stored.

    Returns:
        pd.DataFrame:
            Feature dataset ready for transformation and prediction.
    """

    logger.info("Loading feature data for prediction.")

    feature_data = pd.read_csv(
        data_path(filename, subdir)
    )

    if feature_data.empty:
        logger.error("Feature dataset is empty.")
        raise ValueError("Feature dataset is empty.")

    logger.info(
        f"Feature data loaded successfully. Shape: {feature_data.shape}"
    )

    return feature_data


def run_model():
    """
    Run the prediction pipeline on unseen feature data.

    The function loads engineered features, applies the saved
    PowerTransformer used during training, loads the trained
    Random Forest model, and returns calorie predictions.

    Returns:
        np.ndarray:
            Predicted calorie values rounded to one decimal place.
    """

    logger.info("Starting prediction pipeline.")

    # Load engineered features created by the feature engineering step.
    X_new = read_feature_data(
        "feature_data",
        FEATURES_DIR
    )

    try:
        # Use the same fitted transformer from training.
        # This prevents data leakage and keeps inference consistent
        # with the model training process.
        logger.info("Loading fitted power transformer.")
        ptransformer = load_model(POWER_TRANSFORMER)

        logger.info("Applying power transformer to feature data.")
        X_new_scaled = ptransformer.transform(X_new)

        logger.info(
            f"Feature transformation complete. "
            f"Transformed shape: {X_new_scaled.shape}"
        )

        # Load trained Random Forest model for inference.
        logger.info("Loading trained Random Forest model.")
        rf_model = load_model(RF_MODEL)

        logger.info("Generating predictions.")
        rf_predictions = rf_model.predict(X_new_scaled).round(1)

        logger.info(
            f"Prediction complete. "
            f"Generated {len(rf_predictions)} predictions."
        )

        return rf_predictions

    except FileNotFoundError as error:
        logger.error(
            f"Required model or transformer file not found: {error}"
        )
        raise

    except ValueError as error:
        logger.error(
            f"Prediction failed due to invalid input data: {error}"
        )
        raise

    except Exception as error:
        logger.error(
            f"Unexpected prediction pipeline failure: {error}"
        )
        raise


if __name__ == "__main__":

    logger.info("predict.py execution started.")

    results = run_model()

    logger.info("predict.py execution completed.")
    
    
    
# result_df = y_test
# result_df['prediction'] = rf_predictions
# result_data = result_df.rename({'Calories': 'actual'}, axis=1).reset_index(drop=True)