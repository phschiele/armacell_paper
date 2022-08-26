import os
import pickle
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from models.helpers import get_metric_func
from models.models import get_gru, get_lstm, get_simple, get_autoarima, get_deep_arma, get_shallow_arma


def run_repeated_benchmark(df: pd.DataFrame, compute: bool, path: str, ts_name: str, repetitions: int, mode: str,
                           metric: str, multiprocessing: bool) -> pd.Series:
    if multiprocessing:
        args = [(df, compute, path, ts_name, f"_{i}", i, mode, metric) for i in range(repetitions)]

        with Pool() as pool:
            res = pool.starmap(run_benchmark, args)
    else:
        res = []
        for i in range(repetitions):
            print(f"REPETITION {i}/{repetitions}")
            single_run = run_benchmark(df, compute, path, ts_name, f"_{i}", i, mode, metric)
            res.append(single_run)
    return pd.concat(res, axis=0)


def run_benchmark(df: pd.DataFrame, compute: bool, path: str, ts_name: str, suffix: str, rep: int, mode: str,
                  metric: str) -> pd.Series:
    multi_predictions_file_name = os.path.join(path, f"predictions/{ts_name}_multi_predictions{suffix}.pickle")
    uni_predictions_file_name = os.path.join(path, f"predictions/{ts_name}_uni_predictions{suffix}.pickle")

    metric_func = get_metric_func(metric)

    n = len(df.index)

    scaler = StandardScaler()

    cutoff = int(0.7 * len(df.index))
    train, test = df.iloc[:cutoff], df.iloc[cutoff:]

    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    print("Fitting ARMA")
    # VARMA does not converge, training individual ARMAs
    arma_predictions = []
    for i in range(len(df.columns)):
        single_ts_train = train_scaled[:, i]
        single_ts_test = test_scaled[:, i]
        arma_predictions.append(get_autoarima(single_ts_train, single_ts_test).to_frame())
    pred_armas = pd.concat(arma_predictions, axis=1)
    pred_armas.name = "ARMAs"

    if mode == "multivariate":
        if compute:
            print("Training DeepARMA")
            pred_DeepARMA = get_deep_arma(train_scaled, test_scaled, n, ts_name, rep)
            print("Training ShallowARMA")
            pred_ShallowARMA = get_shallow_arma(train_scaled, test_scaled, n, ts_name, rep)
            print("Training Shallow LSTM")
            pred_shallow_lstm = get_lstm(train_scaled, test_scaled, n, ts_name, rep, deep=True)
            print("Training Deep LSTM")
            pred_deep_lstm = get_lstm(train_scaled, test_scaled, n, ts_name, rep, deep=True)
            print("Training Shallow GRU")
            pred_shallow_gru = get_gru(train_scaled, test_scaled, n, ts_name, rep, deep=False)
            print("Training Deep GRU")
            pred_deep_gru = get_gru(train_scaled, test_scaled, n, ts_name, rep, deep=True)
            print("Training Shallow Simple")
            pred_shallow_simple = get_simple(train_scaled, test_scaled, n, ts_name, rep, deep=False)
            print("Training Deep Simple")
            pred_deep_simple = get_simple(train_scaled, test_scaled, n, ts_name, rep, deep=True)

            multivariate_predictions = {
                "ARMA": pred_armas,
                "DeepARMA": pred_DeepARMA,
                "ShallowARMA": pred_ShallowARMA,
                "ShallowLSTM": pred_shallow_lstm,
                "DeepLSTM": pred_deep_lstm,
                "ShallowGRU": pred_shallow_gru,
                "DeepGRU": pred_deep_gru,
                "ShallowSimple": pred_shallow_simple,
                "DeepSimple": pred_deep_simple,
            }
            with open(multi_predictions_file_name, "wb") as handle:
                pickle.dump(multivariate_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(multi_predictions_file_name, "rb") as handle:
                multivariate_predictions = pickle.load(handle)

        multi_metric_dict = {k: metric_func(v, test_scaled) for k, v in multivariate_predictions.items()}
        multi_metrics = pd.Series(multi_metric_dict)
        multi_metrics.name = ts_name
        return multi_metrics

    elif mode == "univariate":
        if compute:
            univariate_predictions = {}
            for i, col in enumerate(df.columns):
                single_ts_res = {}

                single_ts_train = train_scaled[:, [i]]
                single_ts_test = test_scaled[:, [i]]
                print(f"Training DeepARMA {i}")
                single_ts_res["DeepARMA"] = get_deep_arma(single_ts_train, single_ts_test, n, col, rep)
                print(f"Training ShallowARMA {i}")
                single_ts_res["ShallowARMA"] = get_shallow_arma(single_ts_train, single_ts_test, n, col, rep)
                print(f"Training Deep LSTM {i}")
                single_ts_res["DeepLSTM"] = get_lstm(single_ts_train, single_ts_test, n, col, rep, deep=True)
                print(f"Training Shallow GRU {i}")
                single_ts_res["ShallowGRU"] = get_gru(single_ts_train, single_ts_test, n, col, rep, deep=False)
                print(f"Training Deep GRU {i}")
                single_ts_res["DeepGRU"] = get_gru(single_ts_train, single_ts_test, n, col, rep, deep=True)
                print(f"Training Shallow Simple {i}")
                single_ts_res["ShallowSimple"] = get_simple(single_ts_train, single_ts_test, n, col, rep, deep=False)
                print(f"Training Deep Simple {i}")
                single_ts_res["DeepSimple"] = get_simple(single_ts_train, single_ts_test, n, col, rep, deep=True)
                single_ts_res["ARMA"] = arma_predictions[i]

                univariate_predictions[col] = single_ts_res

            with open(uni_predictions_file_name, "wb") as handle:
                pickle.dump(univariate_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(uni_predictions_file_name, "rb") as handle:
                univariate_predictions = pickle.load(handle)

        metric_dict: Dict[str, Dict[str, float]] = {}
        for time_series, pred_dict in univariate_predictions.items():
            i = list(df.columns).index(time_series)
            metric_dict[time_series] = {}
            for model, predictions in pred_dict.items():
                metric_dict[time_series][model] = metric_func(predictions, test_scaled[:, [i]])

        uni_metrics = pd.DataFrame.from_dict(metric_dict)
        avg_uni_metrics = uni_metrics.mean(axis=1)
        avg_uni_metrics.name = ts_name
        return avg_uni_metrics
    else:
        raise ValueError(mode)


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def create_shifted_frames(data):
    x = data[:, : -1, :, :]
    y = data[:, 1:, :, :]
    return x, y


def create_lagged_shifted_frames(data: np.array, p: int) -> Tuple[np.array, np.array]:
    """
    shift the frames, where `x` is frames 0 to n - 1, and `y` is frames 1 to n.
    """
    x = np.swapaxes(np.swapaxes(np.stack([data[:, i: data.shape[1] - p + i, :, :] for i in range(p)]), 0, 2), 0, 1)
    y = data[:, p:, :, :]
    return x, y


def scale(X: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return (X - lower) / (upper - lower)


def create_sequences(X: np.ndarray, sequence_length: int) -> np.ndarray:
    out = []
    for i in range(int(X.shape[0] - sequence_length)):
        out.append(X[i: i + sequence_length, ...])
    return np.stack(out)
