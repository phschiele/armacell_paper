from typing import Callable

import numpy as np
import pandas as pd


def get_mse(pred: pd.DataFrame, test: np.ndarray) -> float:
    assert pred.shape == test.shape
    return float(np.mean((pred.values - test) ** 2))


def get_mae(pred: pd.DataFrame, test: np.ndarray) -> float:
    assert pred.shape == test.shape
    return float(np.mean(np.abs(pred.values - test)))


def get_rmse(pred: pd.DataFrame, test: np.ndarray) -> float:
    return np.sqrt(get_mse(pred, test))


def get_metric_func(metric: str) -> Callable:
    if metric == "rmse":
        metric_func = get_rmse
    elif metric == "mae":
        metric_func = get_mae
    else:
        raise ValueError(metric)
    return metric_func