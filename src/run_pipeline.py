# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:49:37 2026

@author: dooke
"""
import pandas as pd
from data_preprocessing import upload_clean_data
from paths import PROCESSED_DIR, RESULT_DIR
from utils import data_path
from feature_engineering import upload_features
from predict import run_model


def read_processed_df(filename:str, subdir) -> pd.DataFrame:
    return pd.read_csv(data_path(filename, subdir))

if __name__=="__main__":
    upload_clean_data()
    upload_features()
    output = run_model()
    processed_df = read_processed_df("processed_data", PROCESSED_DIR)
    processed_df["Calories"] = output
    processed_df.to_csv(
        RESULT_DIR / "model_output.csv", 
                               index=False)
    