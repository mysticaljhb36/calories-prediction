# =============================================================================
# Packages Setup
# =============================================================================

import numpy as np
import pandas as pd
from paths import FEATURES_DIR, PROCESSED_DIR
from utils import data_path

# Create module-level logger.
# The logger name automatically matches the module name,
# making it easier to trace pipeline activity.
import logging
logger = logging.getLogger(__name__)


def read_processed_data(filename: str, subdir) -> pd.DataFrame:
    """
    Load processed data ready for feature engineering.

    Args:
        filename (str):
            Name of file to load.

        subdir:
            Directory containing the file.

    Returns:
        pd.DataFrame:
            Processed dataset.
    """

    logger.info("Loading processed dataset.")

    df = pd.read_csv(data_path(filename, subdir))

    logger.info(
        f"Processed dataset loaded successfully. "
        f"Shape: {df.shape}"
    )

    return df


def feature_creation(processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate derived features to improve model performance.

    Feature engineering helps expose non-linear relationships
    between exercise duration, heart rate and calories burned.

    Args:
        processed_data (pd.DataFrame):
            Input dataframe.

    Returns:
        pd.DataFrame:
            Dataframe containing engineered features.
    """

    logger.info("Starting feature engineering.")

    required_columns = {
        "Duration",
        "Pulse",
        "Maxpulse"
    }

    missing_columns = required_columns.difference(
        processed_data.columns
    )
    
    # Test missing columns
    # Raise KeyError if missing columns
    if missing_columns:
        logger.error(
            f"Required columns missing: {missing_columns}"
        )
        raise KeyError(
            f"Required columns missing: {missing_columns}"
        )
    
    # Mathematically validity test for Intensity_Ratio: Pulse/Maxpulse
    # Because Maxpulse is the denominator & Pluse/0 produces inf or NaN 
    # Warn if zero values may cause unrealistic ratios.
    if (processed_data["Maxpulse"] == 0).any():

        logger.warning(
            "Maxpulse contains zero values. "
            "Intensity_Ratio may contain invalid results."
        )
    
    # Data quality test for Pulse
    # Warn if zero or negative values may cause invalid business values.
    if (processed_data["Pulse"] <= 0).any():

        logger.warning(
            "Pulse contains zero or negative values. "
            "Potential sensor or data quality issue detected."
        )
        
    # Data quality test for Duration  
    # Warn if zero or negative values may cause invalid business values.
    if (processed_data["Duration"] <= 0).any():

        logger.warning(
            "Duration contains zero or negative values. "
            "Exercise sessions should normally be positive."
        )
    # Represents cumulative cardiovascular effort
    # over the duration of an exercise session.
    processed_data["Pulse_Duration"] = (
        processed_data["Pulse"] *
        processed_data["Duration"]
    )

    # Captures peak cardiovascular demand
    # sustained throughout the workout.
    processed_data["Maxpulse_Duration"] = (
        processed_data["Maxpulse"] *
        processed_data["Duration"]
    )

    # Normalises exercise intensity relative
    # to the individual's peak heart rate.
    processed_data["Intensity_Ratio"] = (
        processed_data["Pulse"] /
        processed_data["Maxpulse"]
    )

    # Captures interaction between average and
    # peak heart rate measurements.
    processed_data["Pulse_Maxpulse"] = (
        processed_data["Pulse"] *
        processed_data["Maxpulse"]
    )

    # Allows tree and linear models to learn
    # non-linear heart rate relationships.
    processed_data["Pulse_Squared"] = (
        processed_data["Pulse"] ** 2
    )

    # Allows models to learn non-linear
    # exercise duration effects.
    processed_data["Duration_Squared"] = (
        processed_data["Duration"] ** 2
    )

    # Heart Rate Reserve (HRR).
    # Measures available cardiovascular capacity.
    processed_data["Pulse_Reserve"] = (
        processed_data["Maxpulse"] -
        processed_data["Pulse"]
    )

    # Reduces skewness in duration values
    # while preserving ordering.
    processed_data["Log_Duration"] = (
        np.log1p(
            processed_data["Duration"]
        )
    )

    # Approximation of cumulative workload
    # using heart rate and exercise duration.
    processed_data["Heart_Rate_Load"] = (
        processed_data["Duration"] *
        (processed_data["Pulse"] / 100)
    )

    logger.info(
        "Feature engineering complete. "
        "Generated 8 additional features."
    )

    logger.info(
        f"Output dataset shape: "
        f"{processed_data.shape}"
    )

    return processed_data


def upload_features():
    """
    Generate engineered features and save them
    to the features directory.

    Returns:
        None
    """

    logger.info(
        "Starting feature generation pipeline."
    )

    feature_creation(
        read_processed_data(
            "processed_data",
            PROCESSED_DIR
        )
    ).to_csv(
        FEATURES_DIR / "feature_data.csv",
        index=False
    )

    logger.info(
        "Feature dataset saved successfully."
    )


if __name__ == "__main__":

    logger.info(
        "Feature engineering pipeline started."
    )

    upload_features()

    logger.info(
        "Feature engineering pipeline completed."
    )

