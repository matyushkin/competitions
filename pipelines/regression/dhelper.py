# -*- coding: utf-8 -*-
"""Datasets preprocessing.
"""

import os
import urllib

import numpy as np
import pandas as pd

from tqdm import tqdm

from catboost import CatBoostRegressor

def fetch_data(dataset_url, dataset_path):
    """ Fetch dataset files"""
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    if dataset_url.endswith('.tgz'):
        import tarfile
        tgz_path = os.path.join(dataset_path, "dataset.tgz")
        urllib.request.urlretrieve(dataset_url, tgz_path)
        dataset_tgz = tarfile.open(tgz_path)
        dataset_tgz.extractall(path=dataset_path)
        dataset_tgz.close()

def load_data(dataset_path):
    csv_path = os.path.join(dataset_path, "housing.csv")
    return pd.read_csv(csv_path)
        

def smart_imputer(df, target_col):
    '''Fill missing values with prediction model'''
    num_cols = [i for i in df.columns if df[i].dtypes == np.number]
    nan_cols = [i for i in df.columns if df[i].isnull().any()]
    for nan_col in nan_cols:
        if nan_col in num_cols:
            Xy_tmp = df[num_cols].dropna(axis=0)
            d_cols = [nan_col, target_col]
            X_train = Xy_tmp.drop(d_cols, axis=1)
            X_test = df[num_cols][df[nan_col].isnull()].drop(d_cols, axis=1)
            y_train = Xy_tmp[nan_col]
            
            model = CatBoostRegressor(iterations=1e3,
                                      learning_rate=0.1,
                                      depth=10)
            model.fit(X_train,
                      y_train,
                      early_stopping_rounds = 1e3,
                      verbose_eval=False)

            pred = model.predict(X_test)
            df[nan_col][df[nan_col].isnull()] = pred
            nan_cols.remove(nan_col)
    return df
    
    
def make_transformations(X):
    """Make functional transformations for all numeric
    columns of dataset and returns dictionary of
    transformated dataset versions"""
    d = {'x': X,
         'x**0.5': X**0.5,
         'x**2': X**2,
         'log(x)': X.apply(np.log),
         'tanh(x)': X.apply(np.tanh),
         'tanh(ln(x))': X.apply(lambda x: np.tanh(np.log(x))),
         'ln(tanh(x))': X.apply(lambda x: np.log(np.tanh(x))),
         '1/x': 1/X,
         '1/x**0.5': X**(-0.5),
         '1/x**2': X**(-2),
         '1/log(x)': X.apply(lambda x: 1/np.log(x)),
         '1/tanh(x)': X.apply(lambda x: 1/np.tanh(x)),
         '1/tanh(log(x))': X.apply(lambda x: 1/np.tanh(np.log(x))),
         '1/log(tanh(x))': X.apply(lambda x: 1/np.log(np.tanh(x))),
         '1/log(x)**2': X.apply(lambda x: np.power(np.log(x), -2)),
         '1/tanh(x)**2': X.apply(lambda x: np.power(np.tanh(x), -2))}

    for key in d:
        # new column names for every transformation
        start, end = key.split('x')
        d[key].columns = start + d[key].columns + end
    return d


def find_best_compositions(df, df_c):
    num_cols = [i for i in df.columns if df[i].dtypes == np.number]
    
    def make_bool_mask(s):
        return np.array([col in s for col in num_cols])

    df_b = pd.DataFrame(np.stack(df_c.composition.apply(make_bool_mask).values),
                        columns=num_cols,
                        index =df_c.index)

    df_c = pd.concat([df_c, df_b], axis = 1)
    d = df_c.drop(['composition', 'coefficient'], axis=1)
    best_current_row = d.iloc[0].copy()

    print(df_c.iloc[0].composition, df_c.iloc[0].coefficient)
    best_combinations = [df_c.iloc[0].composition]

    for i in tqdm(d.index):
        if np.any(best_current_row & d.loc[i]):
            continue
        else:
            best_current_row = best_current_row | d.loc[i] 
            print(df_c.loc[i].composition, df_c.loc[i].coefficient)
            best_combinations.append(df_c.loc[i].composition)
            if np.all(best_current_row):
                break
                
    return best_combinations
