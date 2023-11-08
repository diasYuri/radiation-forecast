import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

def read_data(filename='../data/daily_barbacena.csv'):
    df_temp = pd.read_csv(
        filename,
        sep=',',
        parse_dates=['Timestamp'],
        index_col=['Timestamp'])

    init_index = lambda _df: _df[_df.RADIATION != 0].index[0]

    df = df_temp[['RADIATION', 'TEMP', 'HUMIDITY_h']] \
        .resample('D') \
        .agg({'RADIATION': 'sum', 'TEMP': 'mean', 'HUMIDITY_h': 'mean'})

    return df.loc[df.index >= init_index(df)].replace(0, np.nan)

def read_data_from_csv(filename, fill=False) -> pd.DataFrame:
    df = pd.read_csv(
        filename,
        sep=',',
        parse_dates=['Timestamp'],
        index_col=['Timestamp'])

    df = df[['RADIATION', 'TEMP', 'HUMIDITY_h']]

    dict_index = {}
    df_temp = df.copy().resample('D').mean()
    for c in df_temp:
        dict_index[c] = df_temp[c].loc[np.isnan(df_temp[c].values)].index
    
    df = df.ffill()\
        .resample('D') \
        .agg({'RADIATION': 'sum', 'TEMP': 'mean', 'HUMIDITY_h': 'mean'})\
        .replace(0, np.nan)

    for c in df:
        df[c].loc[dict_index[c]] = pd.NA

    df = df.loc[df.index >= df[~df.RADIATION.isna()].index[0]]

    if fill:
        return df.interpolate(method='linear', limit_direction='backward')

    return df
    
def windowing_nparray(values, step_back, step_front) -> (np.array, np.array):
    x, y = [], []
    for i in range(len(values) - step_back - step_front):
        j = (i + step_back)
        x.append(values[i:j])
        y.append(values[j:(j+step_front), 0])

    return np.array(x), np.array(y)

def windowing(dataframe, step_back, step_front) -> (np.array, np.array):
    dataset = dataframe.values
    x, y = [], []
    for i in range(len(dataset) - step_back - step_front):
        j = (i + step_back)
        x.append(dataset[i:j])
        y.append(dataset[j:(j+step_front), 0])

    return np.array(x), np.array(y)

def windowing(dataframe, predict_column, step_back, step_front) -> (np.array, np.array):
    dataset = dataframe.values
    dataset_pred = dataframe[predict_column].values

    x, y = [], []
    for i in range(len(dataset) - step_back - step_front):
        j = (i + step_back)
        x.append(dataset[i:j])
        y.append(dataset_pred[j:(j+step_front)])

    return np.array(x), np.array(y)



def split_data(x, y, length, ratio=0.8):
    train_size = int(length * 0.9)
    _train_x, _train_y = x[0:train_size], y[0:train_size].reshape(y[0:train_size].shape[0],)
    _test_x, _test_y = x[train_size:], y[train_size:].reshape(y[train_size:].shape[0],)
    return _train_x, _train_y, _test_x, _test_y


def show_error_metrics(real, pred):
    r2 = r2_score(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = sqrt(mean_squared_error(real, pred))
    mae = mean_absolute_error(real, pred)
    mape = mean_absolute_percentage_error(real, pred)

    print('Test R2: %.3f' % r2)
    print('Test MSE: %.3f' % mse)
    print('Test RMSE: %.3f' % rmse)
    print('Test MAE: %.3f' % mae)
    print('Test MAPE: %.3f' % mape)
