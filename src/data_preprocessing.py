import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def get_scaler():
    scaler = MinMaxScaler(feature_range=(0.1,0.9))
    return scaler


def import_data(path = "../data", fileName = "data.xlsx"):
    raw_data = pd.read_excel(os.path.join(os.path.dirname(__file__), path + "/" + fileName), usecols = ['Date', 'T', 'W', 'SR', 'DSP', 'DRH', 'PanE'])
    df = pd.DataFrame(raw_data)
    return df


def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[(np.abs(zscore(df[numeric_cols])) < 3).all(axis=1)]
    return df

def normalize_df(df):
    scaler = get_scaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
    print('normalized dataframe')
    return scaled_df

# Some of the columns contain the wrong data type so this cleans the data by dropping the cell
def clean_errorenous_data(df):
    df[['T', 'W', 'SR', 'DSP', 'DRH', 'PanE']] = df[['T', 'W', 'SR', 'DSP', 'DRH', 'PanE']].apply(pd.to_numeric, errors='coerce')
    print ('NAN count: ', df.isnull().sum().sum() )
    print (df.dtypes)
    df = df.dropna()
    return df


def denormalize_df(df):
    scaler = get_scaler()
    inverse = scaler.inverse_transform(df)
    inverse = pd.DataFrame(inverse, columns=df.columns)
    return inverse