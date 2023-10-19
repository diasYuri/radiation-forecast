import os

import keras
from keras import layers
from keras_tuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import LSTM
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM, InputLayer
from keras.callbacks import EarlyStopping
import keras


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.LSTM(units=hp.Int('units',
                                       min_value=32,
                                       max_value=512,
                                       step=32),
                          return_sequences=True,
                          activation=hp.Choice('activation_1', ['relu', 'sigmoid']),
                          input_shape=(5, 3)))
    model.add(layers.Dropout(rate=hp.Float(
        'dropout',
        min_value=0.0,
        max_value=0.5,
        default=0.25,
        step=0.05,
    )))
    model.add(layers.LSTM(units=hp.Int('units_2',
                                       min_value=32,
                                       max_value=512,
                                       step=32),
                         activation=hp.Choice('activation_2', ['relu', 'sigmoid'])))
    model.add(layers.Dense(1, activation=hp.Choice('activation_o', ['relu', 'sigmoid'])))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.legacy.Adam(
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])))
    return model


def hp_search(
        name,
        train_x,
        train_y,
        max_trials=10,
        executions_per_trial=1,
        epochs=10,
        validation_split=0.3,
        verbose=0,
        directory='models'
):
    tuner = RandomSearch(
        build_model,
        seed=13418236482,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=name)

    tuner.search_space_summary()

    tuner.search(train_x, train_y,
                 epochs=epochs,
                 verbose=verbose,
                 validation_split=validation_split)

    return tuner.get_best_hyperparameters()[0]


def train_model(
        model_conf,
        train_x,
        train_y,
        name_model,
        epochs=200,
        batch_size=24,
        validation_split=0.2,
        verbose=0,
        use_early_stopping=True,
        early_stopping_patience=15,
        plot_train_history=True,
        use_cache=True,
        cache_path='./trained_models',
        compile_loss='mean_squared_error',
        compile_optimizer=keras.optimizers.legacy.Adam(0.01)
):
    from_cache = use_cache
    loaded_model = None

    if from_cache:
        exist_cache, loaded_model = load_from_cache(name_model, cache_path)
        from_cache = exist_cache

    if from_cache and loaded_model is not None:
        loaded_model.compile(loss=compile_loss, optimizer=compile_optimizer)
        return loaded_model

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=early_stopping_patience)
    model = model_conf
    model.compile(loss=compile_loss, optimizer=compile_optimizer)
    history = model.fit(train_x,
                        train_y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        verbose=verbose,
                        callbacks=[es],
                        workers=4,
                        use_multiprocessing=True)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()

    if use_cache:
        save_trained_model(name_model, model, cache_path)

    return model


def save_trained_model(
        name_model,
        trained_model,
        cache_path):
    path = f'{cache_path}/{name_model}'

    model_path = f'{path}/{name_model}.json'
    weight_path = f'{path}/weight_{name_model}.h5'

    create_path_if_not_exists(path)

    with open(model_path, "w") as json_file:
        json_file.write(trained_model.to_json())

    trained_model.save_weights(weight_path)


def load_from_cache(
        name_model,
        cache_path='./trained_models') -> (bool, any, any):
    path = f'{cache_path}/{name_model}'
    model_path = f'{path}/{name_model}.json'
    weight_path = f'{path}/weight_{name_model}.h5'

    if os.path.isfile(model_path) is False or os.path.isfile(weight_path) is False:
        return False, None

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_path)
    return True, loaded_model


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
