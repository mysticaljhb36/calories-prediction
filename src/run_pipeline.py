# =============================================================================
# Packages Setup
# =============================================================================

import logging

import pandas as pd

from data_preprocessing import upload_processed_data
from feature_engineering import upload_features
from predict import run_model

from paths import PROCESSED_DIR, RESULT_DIR
from utils import data_path
from logger import logger


# Create module-level logger.
# This logger records high-level pipeline execution events.
logger = logging.getLogger(__name__)


def read_processed_df(filename: str, subdir) -> pd.DataFrame:
    """
    Load processed data for merging with model predictions.

    Args:
        filename (str):
            Name of the processed dataset.

        subdir:
            Directory containing the processed dataset.

    Returns:
        pd.DataFrame:
            Processed dataset.
    """

    logger.info(
        "Loading processed dataset for prediction output."
    )

    df = pd.read_csv(
        data_path(filename, subdir)
    )

    logger.info(
        f"Processed dataset loaded successfully. "
        f"Shape: {df.shape}"
    )

    return df


if __name__ == "__main__":

    logger.info(
        "Calories prediction pipeline started."
    )

    try:

        # Stage 1: Data preprocessing
        # Clean raw data and handle missing values before
        # feature engineering and prediction.
        logger.info(
            "Stage 1/3: Running data preprocessing."
        )

        upload_processed_data()

        logger.info(
            "Stage 1/3 completed successfully."
        )

        # Stage 2: Feature engineering
        # Generate derived features expected by the model.
        logger.info(
            "Stage 2/3: Running feature engineering."
        )

        upload_features()

        logger.info(
            "Stage 2/3 completed successfully."
        )

        # Stage 3: Model inference
        # Generate calorie predictions using the trained model.
        logger.info(
            "Stage 3/3: Running prediction."
        )

        predictions = run_model()

        logger.info(
            f"Generated {len(predictions)} predictions."
        )

        # Load processed dataset and append predictions.
        logger.info(
            "Combining predictions with processed data."
        )

        processed_df = read_processed_df(
            "processed_data",
            PROCESSED_DIR
        )

        # Attach predicted calorie values to the dataset.
        processed_df["Calories"] = predictions

        # Validate row counts before saving.
        if len(processed_df) != len(predictions):

            logger.error(
                "Prediction count does not match "
                "processed dataset row count."
            )

            raise ValueError(
                "Prediction and dataset lengths differ."
            )

        output_path = (
            RESULT_DIR / "model_output.csv"
        )

        processed_df.to_csv(
            output_path,
            index=False
        )

        logger.info(
            f"Results saved successfully to "
            f"{output_path}"
        )

        logger.info(
            "Calories prediction pipeline completed "
            "successfully."
        )

    except Exception as error:

        logger.critical(
            f"Pipeline execution failed: {error}"
        )

        raise

