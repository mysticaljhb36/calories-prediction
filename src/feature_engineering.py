# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:16:34 2026

@author: dooke
"""
import pandas as pd
import numpy as np
from paths import FEATURES_DIR, PROCESSED_DIR
from utils import data_path


def read_processed_data(filename:str, subdir) -> pd.DataFrame:
    return pd.read_csv(data_path(filename, subdir))

def feature_creation(processed_data):
    # Pulse × Duration - Total cardiovascular workload
    processed_data['Pulse_Duration'] = (
        processed_data['Pulse'] *
        processed_data['Duration']
    )
    
    # Maxpulse × Duration - Peak intensity sustained over time
    processed_data['Maxpulse_Duration'] = (
        processed_data['Maxpulse'] *
        processed_data['Duration']
    )
    
    # Exercise Intensity Ratio - i.e 0.60 = moderate exercise, 0.85 = vigorous exercise
    processed_data['Intensity_Ratio'] = (
        processed_data['Pulse'] /
        processed_data['Maxpulse']
    )
    
    # Pulse × Maxpulse - Captures combined heart-rate load
    processed_data['Pulse_Maxpulse'] = (
        processed_data['Pulse'] *
        processed_data['Maxpulse']
    )
    
    # Pulse rate
    processed_data['Pulse_Squared'] = (
        processed_data['Pulse'] ** 2
    )
    
    # Duration rate
    processed_data['Duration_Squared'] = (
        processed_data['Duration'] ** 2
    )
    
    # Heart Rate Reserve (HRR) - This captures exercise intensity better than Pulse alone.
    processed_data['Pulse_Reserve'] = (
        processed_data['Maxpulse'] - processed_data['Pulse']
    )
    
    # Log Duration 
    processed_data['Log_Duration'] = np.log1p(
        processed_data['Duration']
    )
    
    # Calories-per-Minute Proxy - Useful interactive feature
    processed_data['Heart_Rate_Load'] = (
        processed_data['Duration'] *
        (processed_data['Pulse'] / 100)
    )
    return processed_data

# Save File
def upload_features():
    return feature_creation(read_processed_data("processed_data", 
                                                         PROCESSED_DIR)
                                     ).to_csv(FEATURES_DIR / "feature_data.csv", index=False)
    

if __name__ == "__main__":
    upload_features()