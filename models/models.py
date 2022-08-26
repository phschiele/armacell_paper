import copy
from typing import Callable, Tuple

import keras_tuner as kt
import numpy as np
import pandas as pd
from keras_tuner import HyperParameters
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense, SimpleRNN
from tensorflow.keras.models import Sequential

from arma_cell.arma import ARMA
from arma_cell.conv_arma import ConvARMA, SpatialDiffs
from arma_cell.helpers import prepare_arma_input


class TrainingSettings:
    EPOCHS = 100
    VERBOSE_FIT = 0
    VERBOSE_TUNER = 0
    PATIENCE = 10
    MAX_TRIALS = 25
    EAGER = False


def get_convarma(n_layers: int, X_shape, Y_shape):
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *X_shape[2:]))

    if n_layers == 1:
        x = SpatialDiffs(5)(inp)
        x = ConvARMA(
            activation="relu",
            units=64,
            q=3,
            image_shape=(*X_shape[2:5], X_shape[-1]*5),
            kernel_size=(3, 3),
            return_lags=False,
            return_sequences=True
        )(x)
    elif n_layers == 2:
        x = SpatialDiffs(5)(inp)
        x = ConvARMA(
            activation="relu",
            units=64,
            q=3,
            image_shape=(*X_shape[2:5], X_shape[-1]*5),
            kernel_size=(3, 3),
            return_lags=True,
            return_sequences=True
        )(x)
        x = layers.BatchNormalization()(x)
        x = ConvARMA(
            activation="relu",
            units=64,
            q=1,
            image_shape=(3, *X_shape[3:5], 64),
            kernel_size=(1, 1),
            return_lags=False,
            return_sequences=True
        )(x)
    elif n_layers == 3:
        x = SpatialDiffs(5)(inp)
        x = ConvARMA(
            activation="relu",
            units=64,
            q=3,
            image_shape=(*X_shape[2:5], X_shape[-1]*5),
            kernel_size=(5, 5),
            return_lags=True,
            return_sequences=True
        )(x)
        x = layers.BatchNormalization()(x)
        x = ConvARMA(
            activation="relu",
            units=64,
            q=1,
            image_shape=(3, *X_shape[3:5], 64),
            kernel_size=(3, 3),
            return_lags=True,
            return_sequences=True
        )(x)
        x = layers.BatchNormalization()(x)
        x = ConvARMA(
            activation="relu",
            units=64,
            q=1,
            image_shape=(1, *X_shape[3:5], 64),
            kernel_size=(1, 1),
            return_lags=False,
            return_sequences=True
        )(x)
    else:
        raise NotImplementedError

    x = layers.Conv2D(
        filters=Y_shape[-1], kernel_size=(3, 3), activation="sigmoid", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = keras.models.Model(inp, x)
    model.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), run_eagerly=True
    )
    print(model.summary())
    return model


def get_convlstm(n_layers: int, X_shape):
    # Construct the input layer with no definite frame size.
    inp = layers.Input(shape=(None, *X_shape[2:]))

    if n_layers == 1:
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
    elif n_layers == 2:
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
    elif n_layers == 3:
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(5, 5),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(inp)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(1, 1),
            padding="same",
            return_sequences=True,
            activation="relu",
        )(x)
    else:
        raise NotImplementedError

    x = layers.Conv2D(
        filters=X_shape[-1], kernel_size=(3, 3), activation="sigmoid", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    lstm = keras.models.Model(inp, x)
    lstm.compile(
        loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
    )
    print(lstm.summary())
    return lstm


def get_varma(train: np.array, test: np.array) -> pd.DataFrame:
    # Note: only testing order (1,1) here.
    model = VARMAX(train, order=(1, 1)).fit(disp=False)
    steps = len(test) + 1
    predictions = np.zeros((steps, test.shape[1]))
    var = model.coefficient_matrices_var[0]
    vma = model.coefficient_matrices_vma[0]

    for i in range(steps):
        if i > 0:
            predictions[i] += var @ test[i - 1]
            predictions[i] += vma @ (-predictions[i - 1] + test[i - 1])

    res = pd.DataFrame(predictions[:-1])
    res.name = "VARMA"
    return res


def get_deep_arma(train: np.array, test: np.array, n: int, ts: str, repetition: int) -> pd.DataFrame:
    split_nn = int(len(train) * 0.7)
    train_nn, val_nn = train[:split_nn], train[split_nn:]
    sequence_length = 10

    n_features = train.shape[1]

    best_vals = {}
    best_hp_dict = {}
    for p in range(1, 5):
        callbacks = [EarlyStopping(monitor="val_loss",
                                   patience=TrainingSettings.PATIENCE)]  # Define in loop as otherwise tuner fails with no deep copy error
        print(f"deep, p = q = {p}", "-" * 25)

        q = p
        X_train, y_train = prepare_arma_input(max(p, q), train_nn, sequence_length=sequence_length)
        X_val, y_val = prepare_arma_input(max(p, q), val_nn, sequence_length=sequence_length)

        def builder(hp: HyperParameters) -> models.Model:
            return deep_arma_builder(p, q, hp, n_features)

        tuner = kt.RandomSearch(
            builder, objective="val_loss", max_trials=TrainingSettings.MAX_TRIALS, overwrite=True,
            directory=f"simulations/tuner/deep_arma_tuner_{n}_{repetition}_{ts}"
        )

        tuner.search(X_train, y_train, epochs=TrainingSettings.EPOCHS, validation_data=(X_val, y_val),
                     verbose=TrainingSettings.VERBOSE_TUNER, batch_size=50, callbacks=callbacks)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"{best_hps.values=}")

        model = tuner.hypermodel.build(best_hps)
        hist = model.fit(X_train, y_train, epochs=TrainingSettings.EPOCHS, validation_data=(X_val, y_val),
                         verbose=TrainingSettings.VERBOSE_FIT, batch_size=50, callbacks=callbacks)
        best_vals[p] = hist.history["val_loss"][-1]
        best_hp_dict[p] = copy.copy(best_hps)

    p = min(best_vals, key=best_vals.get)
    q = p
    X_train, y_train = prepare_arma_input(max(p, q), train_nn, sequence_length=sequence_length)
    X_val, y_val = prepare_arma_input(max(p, q), val_nn)

    def final_builder(hp: HyperParameters) -> models.Model:
        return deep_arma_builder(p, q, hp, n_features)

    model = final_builder(best_hp_dict[p])
    callbacks = [EarlyStopping(monitor="val_loss", patience=TrainingSettings.PATIENCE)]
    model.fit(X_train, y_train, epochs=TrainingSettings.EPOCHS, validation_data=(X_val, y_val),
              verbose=TrainingSettings.VERBOSE_FIT, batch_size=50, callbacks=callbacks)

    X_test, _ = prepare_arma_input(max(p, q), test, sequence_length=sequence_length)
    predictions = model.predict(X_test)
    filled_predictions = np.vstack([np.zeros((len(test) - predictions.shape[0], n_features)), predictions])
    res = pd.DataFrame(filled_predictions)
    res.name = "DeepARMA"
    return res


def get_shallow_arma(train: np.array, test: np.array, n: int, ts: str, repetition: int) -> pd.DataFrame:
    split_nn = int(len(train) * 0.7)
    train_nn, val_nn = train[:split_nn], train[split_nn:]
    sequence_length = 10

    n_features = train.shape[1]

    best_vals = {}
    best_hp_dict = {}
    for p in range(1, 5):
        callbacks = [EarlyStopping(monitor="val_loss",
                                   patience=TrainingSettings.PATIENCE)]  # Define in loop as otherwise tuner fails with no deep copy error
        print(f"shallow, p = q = {p}", "-" * 25)

        q = p
        X_train, y_train = prepare_arma_input(max(p, q), train_nn, sequence_length=sequence_length)
        X_val, y_val = prepare_arma_input(max(p, q), val_nn, sequence_length=sequence_length)

        def builder(hp: HyperParameters) -> models.Model:
            return shallow_arma_builder(p, q, sequence_length, hp, n_features)

        tuner = kt.RandomSearch(
            builder, objective="val_loss", max_trials=TrainingSettings.MAX_TRIALS, overwrite=True,
            directory=f"simulations/tuner/shallow_arma_tuner_{n}_{repetition}_{ts}"
        )

        tuner.search(X_train, y_train, epochs=TrainingSettings.EPOCHS, validation_data=(X_val, y_val),
                     verbose=TrainingSettings.VERBOSE_TUNER, batch_size=50, callbacks=callbacks)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"{best_hps.values=}")

        model = tuner.hypermodel.build(best_hps)
        hist = model.fit(X_train, y_train, epochs=TrainingSettings.EPOCHS, validation_data=(X_val, y_val),
                         verbose=TrainingSettings.VERBOSE_FIT, batch_size=50, callbacks=callbacks)
        best_vals[p] = hist.history["val_loss"][-1]
        best_hp_dict[p] = copy.copy(best_hps)

    p = min(best_vals, key=best_vals.get)
    q = p
    X_train, y_train = prepare_arma_input(max(p, q), train_nn, sequence_length=sequence_length)
    X_val, y_val = prepare_arma_input(max(p, q), val_nn)

    def final_builder(hp: HyperParameters) -> models.Model:
        return shallow_arma_builder(p, q, sequence_length, hp, n_features)

    model = final_builder(best_hp_dict[p])
    callbacks = [EarlyStopping(monitor="val_loss", patience=TrainingSettings.PATIENCE)]
    model.fit(X_train, y_train, epochs=TrainingSettings.EPOCHS, validation_data=(X_val, y_val),
              verbose=TrainingSettings.VERBOSE_FIT, batch_size=50, callbacks=callbacks)

    X_test, _ = prepare_arma_input(max(p, q), test, sequence_length=sequence_length)
    predictions = model.predict(X_test)
    filled_predictions = np.vstack([np.zeros((len(test) - predictions.shape[0], n_features)), predictions])
    res = pd.DataFrame(filled_predictions)
    res.name = "ShallowARMA"
    return res


def get_nn(train: np.array, test: np.array, n: int, ts: str, repetition: int, deep: bool, nn_builder: Callable,
           name: str) -> pd.DataFrame:
    full_name = f"Deep{name}" if deep else name
    split_nn = int(len(train) * 0.7)
    train_nn, val_nn = train[:split_nn], train[split_nn:]
    sequence_length = 10
    callbacks = [EarlyStopping(monitor="val_loss", patience=TrainingSettings.PATIENCE)]

    X_train, y_train = split_sequence(train_nn, sequence_length)
    X_val, y_val = split_sequence(val_nn, sequence_length)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = train.shape[1]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], n_features))

    # define model
    def builder(hp: HyperParameters) -> models.Model:
        return nn_builder(sequence_length, hp, n_features, deep)

    tuner = kt.RandomSearch(
        builder, objective="val_loss", max_trials=TrainingSettings.MAX_TRIALS, overwrite=True,
        directory=f"simulations/tuner/{full_name}_tuner_{n}_{repetition}_{ts}"
    )

    tuner.search(X_train, y_train, epochs=TrainingSettings.EPOCHS, validation_data=(X_val, y_val),
                 verbose=TrainingSettings.VERBOSE_TUNER, batch_size=50, callbacks=callbacks)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"{best_hps.values=}")
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, epochs=TrainingSettings.EPOCHS, validation_data=(X_val, y_val),
              verbose=TrainingSettings.VERBOSE_FIT, batch_size=50, callbacks=callbacks)

    X_test, _ = split_sequence(test, sequence_length)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    filled_predictions = np.vstack([np.zeros((sequence_length, n_features)), model.predict(X_test)])
    res = pd.DataFrame(filled_predictions)
    res.name = full_name
    return res


def get_lstm(train: np.array, test: np.array, n: int, ts: str, repetition: int, deep: bool) -> pd.DataFrame:
    return get_nn(train, test, n, ts, repetition, deep, lstm_builder, "LSTM")


def get_gru(train: np.array, test: np.array, n: int, ts: str, repetition: int, deep: bool) -> pd.DataFrame:
    return get_nn(train, test, n, ts, repetition, deep, gru_builder, "GRU")


def get_simple(train: np.array, test: np.array, n: int, ts: str, repetition: int, deep: bool) -> pd.DataFrame:
    return get_nn(train, test, n, ts, repetition, deep, simple_builder, "SIMPLE")


def get_autoarima(train: np.array, test: np.array) -> pd.Series:
    arma = auto_arima(train, seasonal=False)
    order = arma.get_params()["order"]
    ts_arima = ARIMA(endog=train, order=order).fit()
    steps = len(test) + 1
    predictions = np.zeros(steps)
    ar = ts_arima.arparams if "ar" in ts_arima.param_terms else np.array([])
    ma = ts_arima.maparams if "ma" in ts_arima.param_terms else np.array([])

    for i in range(steps):
        for j, ar_par in enumerate(ar):
            if i > j:
                predictions[i] += test[i - j - 1] * ar_par
        for j, ma_par in enumerate(ma):
            if i > j:
                predictions[i] += (-predictions[i - j - 1] + test[i - j - 1]) * ma_par

    return pd.Series(predictions[:-1], name="ARMA")


# split a univariate sequence into samples
def split_sequence(sequence: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    # taken from https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    X, y = [], []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def shallow_arma_builder(p: int, q: int, sequence_length: int, hp: HyperParameters,
                         n_features: int = 1) -> models.Model:
    input = layers.Input(shape=(sequence_length, n_features, p))
    nonlinear_units = hp.Int(name="nonlinear_units", min_value=0, max_value=5, default=1)
    print(f"{nonlinear_units=}")
    linear_arma = ARMA(q, input_dim=(n_features, p), units=1, activation="linear", use_bias=True)(input)
    if nonlinear_units == 0:
        flat = layers.Flatten()(linear_arma)
        model = models.Model(inputs=input, outputs=flat)
    else:
        nonlinear_arma = ARMA(q, input_dim=(n_features, p), units=nonlinear_units, activation="relu", use_bias=True)(input)
        concat = layers.Concatenate(axis=-2)([linear_arma, nonlinear_arma])
        flat = layers.Flatten()(concat)
        output = layers.Dense(
            n_features,
        )(flat)
        model = models.Model(inputs=input, outputs=output)
    model.compile(loss="mse", optimizer="adam", run_eagerly=TrainingSettings.EAGER)
    return model


def deep_arma_builder(p: int, q: int, hp: HyperParameters, n_features: int = 1) -> models.Model:
    units = hp.Int(name="units", min_value=1, max_value=4, default=1)
    print(f"{units=}")

    ARMA_1 = ARMA(q, input_dim=(n_features, p), use_bias=True, units=units, return_lags=True, activation="relu",
                      name="ARMA_1", return_sequences=True)
    ARMA_2 = ARMA(q, input_dim=(int(units * n_features), q), use_bias=True, units=units, activation="relu",
                      name="ARMA_2")

    model = Sequential(
        [
            ARMA_1,
            ARMA_2,
            layers.Flatten(),
            layers.Dense(n_features, activation=None),
        ]
    )
    model.compile(loss="mse", optimizer="adam", run_eagerly=TrainingSettings.EAGER)
    return model


def lstm_builder(sequence_length: int, hp: HyperParameters, n_features: int, deep: bool) -> models.Model:
    units = hp.Int(name="lstm_units", min_value=1, max_value=5, default=1)
    model = Sequential()
    if deep:
        model.add(LSTM(units, activation="relu", input_shape=(sequence_length, n_features), return_sequences=True))
        model.add(LSTM(units, activation="relu", input_shape=(sequence_length, n_features)))
    else:
        model.add(LSTM(units, activation="relu", input_shape=(sequence_length, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer="adam", loss="mse")
    return model


def gru_builder(sequence_length: int, hp: HyperParameters, n_features: int, deep: bool) -> models.Model:
    units = hp.Int(name="lstm_units", min_value=1, max_value=5, default=1)
    model = Sequential()
    if deep:
        model.add(GRU(units, activation="relu", input_shape=(sequence_length, n_features), return_sequences=True))
        model.add(GRU(units, activation="relu", input_shape=(sequence_length, n_features)))
    else:
        model.add(GRU(units, activation="relu", input_shape=(sequence_length, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer="adam", loss="mse")
    return model


def simple_builder(sequence_length: int, hp: HyperParameters, n_features: int, deep: bool) -> models.Model:
    units = hp.Int(name="lstm_units", min_value=1, max_value=5, default=1)
    model = Sequential()
    if deep:
        model.add(SimpleRNN(units, activation="relu", input_shape=(sequence_length, n_features), return_sequences=True))
        model.add(SimpleRNN(units, activation="relu", input_shape=(sequence_length, n_features)))
    else:
        model.add(SimpleRNN(units, activation="relu", input_shape=(sequence_length, n_features)))
    model.add(Dense(n_features))
    model.compile(optimizer="adam", loss="mse")
    return model
