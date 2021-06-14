import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def import_data(path = "../data", fileName = "data.xlsx"):
    raw_data = pd.read_excel(os.path.join(os.path.dirname(__file__), path + "/" + fileName), usecols = ['Date', 'T', 'W', 'SR', 'DSP', 'DRH', 'PanE'])
    df = pd.DataFrame(raw_data)
    return df


def remove_outliers(df):
    df = df[(np.abs(zscore(df.columns)) < 3).all(axis=1)]
    return df

def normalize_df(df):
    scaler = MinMaxScaler(feature_range=(0.1,0.9))
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns[1:])
    return scaled_df