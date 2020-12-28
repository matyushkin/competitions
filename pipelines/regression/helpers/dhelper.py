# -*- coding: utf-8 -*-
"""Datasets preprocessing.
"""

import os
import urllib

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from tqdm import tqdm

from catboost import CatBoostRegressor


# ---- FETCH AND LOAD DATA ----

def fetch_data(dataset_url, dataset_path):
    """ Fetches dataset file from dataset_url
    and saves them to dataset_path folder"""
    import urllib
    file_name = dataset_url.split('/')[-1]
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    if file_name.endswith('.tgz'):
        import tarfile
        tgz_path = os.path.join(dataset_path, file_name)
        urllib.request.urlretrieve(dataset_url, tgz_path)
        dataset_tgz = tarfile.open(tgz_path)
        dataset_tgz.extractall(path=dataset_path)
        dataset_tgz.close()
        os.remove(os.path.join(dataset_path, file_name))
    elif file_name.endswith('.csv'):
        csv_path = os.path.join(dataset_path, file_name)
        urllib.request.urlretrieve(dataset_url, csv_path)


def load_data(dataset_path):
    """Load data from dataset_path file. In the case
    when there are a lot of files in dataset_path directory
    it loads the largest one"""
    csv_filenames = [i for i in os.listdir(dataset_path) if i.endswith('.csv')]
    d = dict()
    for csv_filename in csv_filenames:
        filepath = os.path.join(dataset_path, csv_filename)
        d[csv_filename] = os.stat(filepath).st_size
    df_filename = max(d, key=d.get)
    csv_path = os.path.join(dataset_path, df_filename)
    return pd.read_csv(csv_path)


def current_cols_status(X_num, X_cat):
    num_cols = X_num.columns
    cat_cols = X_cat.columns
    nan_num_cols = [col for col in num_cols if X_num[col].isnull().any()]
    nan_cat_cols = [col for col in cat_cols if X_cat[col].isnull().any()]
    v = {'numerical columns': num_cols,
         'numerical columns with N/A values': nan_num_cols,
         'categorical columns': cat_cols,
         'categorical columns with N/A values': nan_cat_cols}
    
    for key in v:
        print(f'{key}:')
        for item in v[key]:
            print(f' - {item}')
        if not list(v[key]):
            print(f' ----')



def simple_imputer(df):
    num_cols = [i for i in df.columns if is_numeric_dtype(df[i])]
    nan_cols = [i for i in df.columns if df[i].isnull().any()]
    for nan_col in nan_cols:
        if nan_col in num_cols:
            median = df[nan_col].median()
            df[nan_col].fillna(median, inplace=True)
    return df
   

def smart_imputer(X_num):
    '''Fill missing values with simple prediction model'''
    nan_cols = [col for col in X_num if X_num[col].isnull().any()]
    for nan_col in nan_cols:
        condition = X_num[nan_col].isnull()
        X_test = X_num[condition].drop(nan_col, axis=1)
        X_train = X_num[~condition].drop(nan_col, axis=1)
        X_train.dropna(axis=0, inplace=True)
        y_train = X_num.loc[~condition, nan_col]
        model = CatBoostRegressor(iterations=1e3,
                                  learning_rate=0.1,
                                  depth=10)
        model.fit(X_train,
                  y_train,
                  verbose_eval=False)

        y_pred = model.predict(X_test)
        X_num.loc[condition, nan_col] = y_pred
    return X_num
    
    
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
