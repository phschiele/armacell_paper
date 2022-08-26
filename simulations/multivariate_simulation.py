from multiprocessing import Pool
from typing import Tuple

import fire
import numpy as np
import pandas as pd
from arma_cell.helpers import set_all_seeds, simulate_varma_process
from models.helpers import get_rmse, get_mae
from simulations.helpers import EXP, SQ

from models.models import get_deep_arma, get_shallow_arma, get_lstm, get_varma, get_gru, get_simple


def run_single_multivariate_ts(ts: pd.DataFrame, ts_name: str, repetition: int) -> pd.DataFrame:
    n = len(ts.index)
    split = int(n * 0.7)
    train, test = ts[:split].values, ts[split:].values

    pred_DeepARMA = get_deep_arma(train, test, n, ts_name, repetition)
    pred_DeepARMA.columns = [f"{pred_DeepARMA.name}{c}" for c in pred_DeepARMA.columns]

    pred_ShallowARMA = get_shallow_arma(train, test, n, ts_name, repetition)
    pred_ShallowARMA.columns = [f"{pred_ShallowARMA.name}{c}" for c in pred_ShallowARMA.columns]

    pred_lstm = get_lstm(train, test, n, ts_name, repetition, deep=False)
    pred_lstm.columns = [f"{pred_lstm.name}{c}" for c in pred_lstm.columns]

    pred_deep_lstm = get_lstm(train, test, n, ts_name, repetition, deep=True)
    pred_deep_lstm.columns = [f"{pred_deep_lstm.name}{c}" for c in pred_deep_lstm.columns]

    pred_shallow_gru = get_gru(train, test, n, ts_name, repetition, deep=False)
    pred_shallow_gru.columns = [f"{pred_shallow_gru.name}{c}" for c in pred_shallow_gru.columns]

    pred_deep_gru = get_gru(train, test, n, ts_name, repetition, deep=False)
    pred_deep_gru.columns = [f"{pred_deep_gru.name}{c}" for c in pred_deep_gru.columns]

    pred_shallow_simple = get_simple(train, test, n, ts_name, repetition, deep=False)
    pred_shallow_simple.columns = [f"{pred_shallow_simple.name}{c}" for c in pred_shallow_simple.columns]

    pred_deep_simple = get_simple(train, test, n, ts_name, repetition, deep=False)
    pred_deep_simple.columns = [f"{pred_deep_simple.name}{c}" for c in pred_deep_simple.columns]

    pred_varma = get_varma(train, test)
    pred_varma.columns = [f"{pred_varma.name}{c}" for c in pred_varma.columns]

    y = pd.DataFrame(test)
    y.name = ts_name
    y.columns = [f"{y.name}{c}" for c in y.columns]

    preds = pd.concat([pred_varma, pred_DeepARMA, pred_ShallowARMA, pred_lstm, pred_deep_lstm, pred_shallow_gru, pred_deep_gru, pred_shallow_simple, pred_deep_simple, y], axis=1)

    return preds


def single_rep(repetition: int, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_all_seeds(repetition)
    VAR = np.array([[0.1, -0.2], [-0.2, 0.1]])
    VAR = np.expand_dims(VAR, axis=-1)
    VMA = np.array([[-0.4, 0.2], [0.2, -0.4]])
    VMA = np.expand_dims(VMA, axis=-1)
    alpha = np.zeros(2)
    Y1 = pd.DataFrame(simulate_varma_process(VAR, VMA, alpha, n_steps=n))
    Y1.name = "VARMA"
    Y2 = pd.DataFrame(EXP(n))
    Y2.name = "EXP"
    Y3 = pd.DataFrame(SQ(n))
    Y3.name = "SQ"

    single_ts = [Y1, Y2, Y3]
    single_ts = [(Y, Y.name) for Y in single_ts]

    rmse = []
    mae = []
    for single_t in single_ts:
        predictions = run_single_multivariate_ts(single_t[0], single_t[1], repetition)
        rmse_metrics = pd.Series(
            {
                predictions.columns[i * 2][:-1]: (
                    get_rmse(predictions.iloc[:, i * 2: i * 2 + 2], predictions.iloc[:, -2:].values))
                for i in range(len(predictions.columns) // 2 - 1)
            },
            name=single_t[1],
        ).to_frame()
        rmse.append(rmse_metrics)

        mae_metrics = pd.Series(
            {
                predictions.columns[i * 2][:-1]: (
                    get_mae(predictions.iloc[:, i * 2: i * 2 + 2], predictions.iloc[:, -2:].values))
                for i in range(len(predictions.columns) // 2 - 1)
            },
            name=single_t[1],
        ).to_frame()
        mae.append(mae_metrics)

    rmse = pd.concat(rmse, axis=1)
    mae = pd.concat(mae, axis=1)
    return rmse, mae


def run(n: int = 1000, multiprocessing: bool = False, repetitions: int = 1) -> None:
    if multiprocessing:
        args = [(i, n) for i in range(repetitions)]
        with Pool() as pool:
            res = pool.starmap(single_rep, args)
    else:
        res = []
        for repetition in range(repetitions):
            print("#" * 30)
            print("#" * 9 + f"{repetition=}" + "#" * 9)
            print("#" * 30)
            res.append(single_rep(repetition, n))

    df_rmse = pd.concat([r[0] for r in res], axis=0)
    df_rmse.to_csv("simulations/results/multivariate_simulation_rmse.csv")
    df_mae = pd.concat([r[1] for r in res], axis=0)
    df_mae.to_csv("simulations/results/multivariate_simulation_mae.csv")


if __name__ == "__main__":
    # Usage:
    # python simulations/multivariate_simulation.py
    fire.Fire(run)
