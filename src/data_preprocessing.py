# =============================================================================
# Packages Setup
# =============================================================================
import pandas as pd


# Import impulse data from raw subdirectory
from paths import PROCESSED_DIR, UNSEEN_DIR
from utils import transform_imputer, data_path


@transform_imputer
def process_data_file(filename:str, subdir) -> pd.DataFrame:
    return pd.read_csv(data_path(filename, subdir))

def upload_clean_data():
    return process_data_file("unseen_data", UNSEEN_DIR).to_csv(
        PROCESSED_DIR / "processed_data.csv", 
                               index=False) # index=False prevents row numbers
    

if __name__ == "__main__":
    # Save File    
    upload_clean_data()
    


