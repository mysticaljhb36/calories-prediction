# =============================================================================
# Packages Setup
# =============================================================================
import pandas as pd
import logging
# To create or retrieve a python logger object
logger = logging.getLogger(__name__)

from paths import PROCESSED_DIR, UNSEEN_DIR
from utils import transform_imputer, data_path

# =============================================================================
# Load Unseen Data
# =============================================================================
# Read unseen data that will be preprocessed prior to prediction.
read_data = pd.read_csv(data_path("unseen_data", UNSEEN_DIR))


logger.debug("Raw dataset shape: %s", read_data)
# =============================================================================
# Imputation Pipeline
# =============================================================================
@transform_imputer
def apply_imputer(x_data):
    """
    Apply iterative imputation to the supplied dataset.

    The actual imputation logic is handled by the
    transform_imputer decorator.
    """
    return x_data


def impute_if_needed(df):
    """
    Check for missing values and apply imputation only when required.

    Args:
        df (pd.DataFrame):
            Input dataframe to validate.

    Returns:
        pd.DataFrame:
            Original dataframe if no missing values exist,
            otherwise the imputed dataframe.
    """

    # Count total missing values across the entire dataframe
    before = df.isna().sum().sum()

    if before > 0:

        logger.info("Checking missing values")

        logger.warning(
            f"{before} missing values detected."
        )

        # Apply iterative imputation
        df = apply_imputer(df)

        # Validate imputation result
        after = df.isna().sum().sum()

        logger.info(
            f"Imputation complete. Missing values reduced "
            f"from {before} to {after}."
        )

    else:

        logger.info(
            "No missing values detected."
        )

    return df


# =============================================================================
# Export Processed Data
# =============================================================================
def upload_processed_data():
    """
    Run preprocessing and save the cleaned dataset.

    Returns:
        None
    """

    return (
        impute_if_needed(read_data)
        .round(1)
        .to_csv(
            PROCESSED_DIR / "processed_data.csv",
            index=False  # Prevent pandas index from being written to file
        )
    )


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":

    logger.info("Starting preprocessing pipeline.")

    # Process and save cleaned dataset
    upload_processed_data()

    logger.info("Preprocessing pipeline completed.")