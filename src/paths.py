from pathlib import Path
import pandas as pd




# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

print(Path(__file__).resolve())

# Data directories
REPORT_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
UNSEEN_DIR = DATA_DIR / "new_data"
RESULT_DIR = DATA_DIR / "result"

MODELS_DIR = PROJECT_ROOT / "models"
RF_MODEL = MODELS_DIR / "random_forest_model.pkl"
POWER_TRANSFORMER = MODELS_DIR / "power_transformer.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# for path in [RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
#     path.mkdir(parents=True, exist_ok=True)

