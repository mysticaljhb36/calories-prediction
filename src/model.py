# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:40:49 2026

@author: dooke
"""

import pandas as pd
from utils import load_model
from paths import POWER_TRANSFORMER, RF_MODEL
# Import impulse data from raw subdirectory
from paths import dataload_csv, FEATURES_DIR



def read_feature_data(filename:str, subdir) -> pd.DataFrame:
    return pd.read_csv(dataload_csv(filename, subdir))


def run_model():
    # load unseen data
    X_new = read_feature_data("feature_data", FEATURES_DIR)
    
    # Import power transformer
    ptransformer = load_model(POWER_TRANSFORMER)
    X_new_scaled = ptransformer.transform(X_new)

    # Import random forest model
    rf_model = load_model(RF_MODEL)

    rf_predictions = rf_model.predict(X_new_scaled).round(1)
    return rf_predictions



if __name__ == "__main__":
    results = run_model()
    
    
    
# result_df = y_test
# result_df['prediction'] = rf_predictions
# result_data = result_df.rename({'Calories': 'actual'}, axis=1).reset_index(drop=True)