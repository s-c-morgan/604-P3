import pandas as pd
import numpy as np
import itertools
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm # For progression bar

def df_to_dict(df, label_attrs, record_attr, save_as_list=False):
    '''
    Convert a pandas.DataFrame object to nested dictionary
    '''
    output = dict()
    for index, row in df.iterrows():
        temp = output
        for label in label_attrs[:-1]:
            temp = temp.setdefault(row[label], dict())
        if save_as_list:temp.setdefault(row[label_attrs[-1]], []).append(row[record_attr])
        else: temp[row[label_attrs[-1]]] = row[record_attr]
    return output

def dfs(tree):
    '''
    Iterate through the nested dictionary-structure
    '''
    if type(tree) == dict:
        for key, value in tree.items():
            for tail in dfs(value):
                yield [key]+tail
    elif type(tree) == list:
        for value in tree: yield [value]
    else: yield [tree]

def dict_to_df(dictionary, label_attrs, record_attr):
    '''
    Convert a nested dictionary to pandas.DataFrame object
    '''
    output = pd.DataFrame(columns=label_attrs+[record_attr])
    for row in dfs(dictionary):
        output.loc[len(output)] = row
    return output

def calculate_delta(df, label_attrs, time_attr, record_attr, n, dif_name=None):
    '''
    Output a pandas.DataFrame that records the difference in `time_attr`
    '''
    if dif_name == None: dif_name = record_attr + '_delta_'
    output = pd.DataFrame(columns=label_attrs+[time_attr]+[dif_name])
    data = df_to_dict(df, label_attrs+[time_attr], record_attr, save_as_list=False)
    unique_values = [list(df[label].unique())for label in label_attrs]
    all_combinations = list(itertools.product(*unique_values))
    for combination in all_combinations:
        temp = data
        for name in combination: temp = temp[name]
        for i in range(n-1):
            delta = temp[i+1]-temp[i]
            output.loc[len(output)] = list(combination)+[i+1]+[delta]
    return output

if __name__ == '__main__':
    data = {"Label1": [1, 1, 1, 1, 2, 2, 2, 2],
            "Label2": [1, 1, 2, 2, 1, 1, 2, 2],
            "Time": [0, 1, 0, 1, 0, 1, 0, 1], 
            "Val": [3, 7, 5, 8, 6, 7, 5, 4]}
    print(calculate_delta(pd.DataFrame(data), ["Label1", "Label2"], "Time", "Val", 2, "Dif"))
    