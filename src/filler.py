import pandas as pd
import numpy as np
from pydmd import DMD, HODMD
from sklearn.ensemble import RandomForestRegressor 
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM, InputLayer
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

def fill_df(dataframe, filler):
    data = {}
    for column in dataframe.columns:
        data[column] = filler(dataframe[column])
    return pd.DataFrame(data=data, index=dataframe.index.values)


def interpolate_filler(data):
    return data.interpolate(method='linear', limit_direction='backward')



def deprecated_dmd_filler(data: pd.Series):
    column = data.name
    data = data.to_frame()
    filled_data = data.copy()
    matrix_data = filled_data.copy().values.T

    col_mean = np.nanmean(matrix_data, axis=1)
    inds_nan = np.where(np.isnan(matrix_data))
    matrix_data[inds_nan] = np.take(col_mean, inds_nan[0])

    dmd = DMD(svd_rank=matrix_data.shape[0])
    dmd.fit(matrix_data)
    dmd_data = dmd.reconstructed_data.real.T
    filled_data.values[np.isnan(data.values)] = dmd_data[np.isnan(data.values)]

    return filled_data[column]

def seasonal_filler(df: pd.Series, period=365, factor=1):
    data = df
    data_temp = data.copy()
    data_temp.interpolate(method='linear', limit_direction='backward', inplace=True)
    data_temp.ffill(inplace=True)
    data_temp.bfill(inplace=True)

    decomposition = seasonal_decompose(data_temp, period=period, model='additive', extrapolate_trend='freq')

    seasonal_component = decomposition.seasonal
    seasonal_component = abs(seasonal_component * factor)

    data_interp = data
    print(df.isna().sum())
    data_interp.loc[data.isna()] = seasonal_component.loc[data.isna()]

    return data_interp


class FillerHelper:
    @staticmethod
    def get_largest_complete_interval(data: pd.Series):
        init, end = 0, 0
        largest_gap = 0
        temp_init = 0

        is_complete = False
        
        for i, r in enumerate(data):
            if pd.isna(r) or r is None:
                if is_complete == True:
                    is_complete = False
                    if (i -1 - temp_init) > largest_gap:
                        largest_gap = i - 1 - temp_init
                        init = temp_init
                        end = i - 1

            else:
                if is_complete == False:
                    is_complete = True
                    temp_init = i

        return data.iloc[init:end]


    @staticmethod
    def find_largest_complete_interval_final(data: pd.Series):
        complete_rows = data.notna().all(axis=1).astype(int) 

        diff = complete_rows.diff()
        diff.iloc[0] = complete_rows.iloc[0]

        starts = diff[diff == 1].index.values[:-1]
        ends = diff[diff == -1].index.values

        if len(starts) > len(ends):
            ends = np.append(ends, complete_rows.index[-1])

        lengths = ends - starts
        max_length_index = lengths.argmax()

        return starts[max_length_index], ends[max_length_index] - 1, lengths[max_length_index]

    @staticmethod
    def extract_largest_complete_interval(data):
        start, end, _ = FillerHelper.find_largest_complete_interval_final(data)
        return data.loc[start:end]
    
    @staticmethod
    def introduce_gaps_dataframe(data, missing_percentage=0.1, random_seed=42, min_gap_size=5, max_gap_size=20):
        np.random.seed(random_seed)
        data_with_gaps = data.copy()

        for column in data.columns:
            num_missing = int(missing_percentage * len(data))
            missing_indices = np.random.choice(len(data), size=num_missing, replace=False)

            for idx in missing_indices:
                gap_size = np.random.randint(min_gap_size, max_gap_size)
                data_with_gaps[column].iloc[(int)(idx):(int)(idx+gap_size)] = pd.NA

        return data_with_gaps
    
    @staticmethod
    def introduce_gaps(data: pd.Series, missing_percentage=0.1, random_seed=42, min_gap_size=5, max_gap_size=20):
        np.random.seed(random_seed)
        data_with_gaps = data.copy()

        num_missing = int(missing_percentage * len(data))
        missing_indices = np.random.choice(len(data), size=num_missing, replace=False)

        for idx in missing_indices:
            gap_size = np.random.randint(min_gap_size, max_gap_size)
            data_with_gaps.iloc[(int)(idx):(int)(idx+gap_size)] = pd.NA

        return data_with_gaps

class Debbuger:
    @staticmethod
    def log(is_debug, *values: object):
        if is_debug:
            print(values)
            
class RandomForestFiller:
    model: RandomForestRegressor
    n_in: int = 5
    n_out: int = 1
    is_debug: bool = False
    random_state: int = 97423897

    def __init__(self, n_estimators, n_in=5, n_out=1, history_length=200, fill_without_retrain=30):
        self.n_estimators = n_estimators
        self.n_in = n_in
        self.n_out = n_out
        self.history_length = history_length
        self.fill_without_retrain = fill_without_retrain

    def __windowing(self, values, step_back, step_front):
        x, y = [], []
        for i in range(len(values) - step_back - step_front + 1):
            j = (i + step_back)
            x.append(values[i:j])
            y.append(values[j:(j+step_front)])
        return np.array(x), np.array(y)

    def __train(self, history):
        Debbuger.log(self.is_debug, 'History:',history)
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        trainX, trainY = self.__windowing(values=history, step_back=self.n_in, step_front=self.n_out)
        Debbuger.log(self.is_debug, 'Train Data:',trainX, trainY)
        self.model.fit(trainX, trainY.ravel())
        

    def __fill(self, history, retrain):
        if retrain:
            self.n_in = min((len(history)-self.n_out), 5)
            self.__train(history=history)
        
        xhat = [history[-self.n_in:]]
        Debbuger.log(self.is_debug, 'xhat', xhat)
        yhat = self.model.predict(xhat)[0]
        Debbuger.log(self.is_debug, 'Predict value:', yhat)
        return yhat


    def __update_history(self, history):
        length = min(200, len(history))
        return history[-length:]

    def filler(self, dataserie: pd.Series):
        dataserie_filled = dataserie.copy()

        history = []

        filling = False
        fill_without_retrain = 0

        for i, r in enumerate(dataserie_filled):
            if pd.isna(r) or r is None:
                history = self.__update_history(history)
                
                retrain = filling is False or fill_without_retrain >= 30
                new_value = self.__fill(history, retrain=retrain) 

                dataserie_filled.iloc[i] = new_value
                history.append(new_value)

                if filling:
                    Debbuger.log(self.is_debug, 'completando...')
                    fill_without_retrain += 1
                else:
                    Debbuger.log(self.is_debug, 'começando a completar...')
                    filling = True
            else:
                history.append(r)
                if filling:
                    Debbuger.log(self.is_debug, 'lacuna completada!')
                    filling = False
                    fill_without_retrain = 0
        return dataserie_filled

           
class RandomForestFillerWithOneModel:
    model: RandomForestRegressor
    n_in: int = 5
    n_out: int = 1
    is_debug: bool = False
    random_state: int = 97423897

    def __init__(self, n_estimators, n_in=5, n_out=1, history_length=200, fill_without_retrain=30):
        self.n_estimators = n_estimators
        self.n_in = n_in
        self.n_out = n_out
        self.history_length = history_length
        self.fill_without_retrain = fill_without_retrain

    def __windowing(self, values, step_back, step_front):
        x, y = [], []
        for i in range(len(values) - step_back - step_front + 1):
            j = (i + step_back)
            x.append(values[i:j])
            y.append(values[j:(j+step_front)])
        return np.array(x), np.array(y)

    def __train(self, history):
        Debbuger.log(self.is_debug, 'History:',history)
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        trainX, trainY = self.__windowing(values=history, step_back=self.n_in, step_front=self.n_out)
        Debbuger.log(self.is_debug, 'Train Data:',trainX, trainY)
        self.model.fit(trainX, trainY.ravel())
        

    def __fill(self, history):
        xhat = [history[-self.n_in:]]
        Debbuger.log(self.is_debug, 'xhat', xhat)
        yhat = self.model.predict(xhat)[0]
        Debbuger.log(self.is_debug, 'Predict value:', yhat)
        return yhat


    def __init_model(self, dataserie: pd.Series):
        train_df = FillerHelper.get_largest_complete_interval(dataserie);
        self.__train(train_df.values)


    def __update_history(self, history):
        length = min(200, len(history))
        return history[-length:]

    def filler(self, dataserie: pd.Series):
        dataserie_filled = dataserie.copy()

        self.__init_model(dataserie_filled)

        history = []

        filling = False
        fill_without_retrain = 0

        for i, r in enumerate(dataserie_filled):
            if pd.isna(r) or r is None:
                history = self.__update_history(history)
                
                new_value = self.__fill(history) 

                dataserie_filled.iloc[i] = new_value
                history.append(new_value)

                if filling:
                    Debbuger.log(self.is_debug, 'completando...')
                    fill_without_retrain += 1
                else:
                    Debbuger.log(self.is_debug, 'começando a completar...')
                    filling = True
            else:
                history.append(r)
                if filling:
                    Debbuger.log(self.is_debug, 'lacuna completada!')
                    filling = False
                    fill_without_retrain = 0
        return dataserie_filled


           
class LstmFillerModel:
    model: Sequential
    n_in: int = 5
    n_out: int = 1
    is_debug: bool = True
    random_state: int = 97423897
    scaler: MinMaxScaler

    def __init__(self, n_in=5, n_out=1, history_length=200, fill_without_retrain=30):
        self.n_in = n_in
        self.n_out = n_out
        self.history_length = history_length
        self.fill_without_retrain = fill_without_retrain

    def __windowing(self, values, step_back, step_front):
        x, y = [], []
        for i in range(len(values) - step_back - step_front):
            j = (i + step_back)
            x.append(values[i:j])
            y.append(values[j:(j+step_front)])
        return np.array(x), np.array(y)

    def __train(self, history):
        Debbuger.log(self.is_debug, 'History:',history)
        
        trainX, trainY = self.__windowing(values=history, step_back=self.n_in, step_front=self.n_out)

        Debbuger.log(self.is_debug, 'Shape trainX:', trainX.shape)
        self.model = Sequential([
            LSTM(320, input_shape=(self.n_in, 1), activation='relu', return_sequences=True),
            Dropout(rate=0.25),
            LSTM(units=180, activation='relu'),
            Dense(1, activation='relu')
        ])

        self.model.compile(loss='mean_squared_error', optimizer=keras.optimizers.legacy.Adam())

        Debbuger.log(self.is_debug, 'Train Data:', trainX, trainY)
        self.model.fit(
                     trainX,
                     trainY,
                     epochs=30,
                     batch_size=32,
                     validation_split=0.1,
                     verbose=0,
                     workers=4,
                     use_multiprocessing=True)
        

    def __fill(self, history):
        print('Length history:', len(history))
        x = np.array(history[-self.n_in:], dtype='object')\
            .astype(float)\
            .reshape(-1,1)
        xhat = self.scaler.transform(x)
        Debbuger.log(self.is_debug, 'xhat', xhat, xhat.shape)
        yhat = self.model.predict(xhat)[0]
        yhat = self.scaler.inverse_transform(yhat.reshape(-1, 1)).reshape(1, -1)[0][0]
        Debbuger.log(self.is_debug, 'Predict value:', yhat)
        return yhat


    def __init_model(self, dataserie: pd.Series):
        train_df = FillerHelper.get_largest_complete_interval(dataserie);
        self.scaler = MinMaxScaler(feature_range=(0,1))
        data = self.scaler.fit_transform(train_df.values.reshape(-1, 1))
        self.__train(data)


    def __update_history(self, history):
        length = min(200, len(history))
        return history[-length:]

    def filler(self, dataserie: pd.Series):
        dataserie_filled = dataserie.copy()

        self.__init_model(dataserie_filled)

        history = []

        filling = False
        fill_without_retrain = 0

        for i, r in enumerate(dataserie_filled):
            if pd.isna(r) or r is None:
                history = self.__update_history(history)
                
                new_value = self.__fill(history) 

                dataserie_filled.iloc[i] = new_value
                history.append(new_value)

                if filling:
                    Debbuger.log(self.is_debug, 'completando...')
                    fill_without_retrain += 1
                else:
                    Debbuger.log(self.is_debug, 'começando a completar...')
                    filling = True
            else:
                history.append(r)
                if filling:
                    Debbuger.log(self.is_debug, 'lacuna completada!')
                    filling = False
                    fill_without_retrain = 0
        return dataserie_filled  


class HodmdFiller:
    def __init__(self, d_factor=0.9):
        self.d_factor=d_factor

    def __reconstruct_data(self, values, steps):
        length = len(values)
        for v in values:
            if pd.isna(v):
                print(values.T)

        hodmd = HODMD(
                    svd_rank=0,
                    tlsq_rank=0,
                    exact=True,
                    opt=True,
                    forward_backward=True,
                    sorted_eigs='abs',
                    rescale_mode='auto',
                    reconstruction_method="mean",
                    d=int(length*self.d_factor)) \
            .fit(values.T)

        hodmd.original_time['tend'] = hodmd.dmd_time['tend'] = length-1
        hodmd.dmd_time['tend'] = length+steps-1

        return hodmd.reconstructed_data.T.real[length:]
    
    def __filler(self, values, empty_gaps):
        data_filled = values

        length_gaps = len(empty_gaps)

        for i in range(length_gaps):
            (idx_lower, idx_top) = empty_gaps[i]

            if i > 0:
                (_, last_idx) = empty_gaps[i - 1]
            else:
                last_idx = 0

            steps = idx_top - idx_lower
            data_input = values[last_idx:idx_lower]

            if len(data_input) < 3 or steps > 15:
                data_input = data_filled[0:idx_lower-1]

            for v in data_input:
                if pd.isna(v):
                    print('dados restaurados:', data_input)

            data = self.__reconstruct_data(data_input, steps)

            for j in range(idx_lower, idx_top):
                data_filled[j] = data[j - idx_lower]

            #print(f'Process in: {(i/length_gaps)*100}%')


        return data_filled           
                    

    def look_for_empty_gaps(self, values):
        empty_spaces = []
        init = None

        for i, value in enumerate(values):
            if pd.isna(value):
                if init is None:
                    init = i
            else:
                if init is not None:
                    empty_spaces.append((init, i))
                    init = None

        if init is not None:
            empty_spaces.append((init, len(values) - 1))

        return empty_spaces
    
    def check_gap(self, idx, dict, list):
        for e in list:
            if idx >= e:
                if idx <= dict[e]:
                    return True    
            else:
                break
        return False

    
    def check(self, values, empty_gaps):
        dict = {}

        for e in empty_gaps:
            (init, end) = e
            dict[init] = end

        list = [i for (i, _) in empty_gaps]
        list.sort()

        for i, value in enumerate(values):
            if pd.isna(value):
                checked = self.check_gap(i, dict=dict, list=list)
                if checked is False:
                    print('Deu ruim:', i)



    def dmd_filler(self,
            serie: pd.Series,
            debug: bool = False) -> pd.Series:
        data = serie.copy()
        values = data.values.reshape(-1,1)
        index = data.index.values

        print(values.shape)

        empty_gaps = self.look_for_empty_gaps(values)
        self.check(data, empty_gaps)

        data_filled = self.__filler(values, empty_gaps)

        serie_filled = pd.Series(data_filled.ravel(),
                        index=index,
                        name=data.name)
        serie_filled.ffill(inplace=True)
        print('DMDFiller -> Dados Faltantes:', serie_filled.isna().sum())
        return serie_filled