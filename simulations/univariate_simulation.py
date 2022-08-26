from multiprocessing import Pool
from typing import Tuple

import fire
import numpy as np
import pandas as pd
from arma_cell.helpers import set_all_seeds, simulate_arma_process
from models.helpers import get_mae, get_rmse
from models.models import get_deep_arma, get_shallow_arma, get_lstm, get_gru, get_simple, get_autoarima
from simulations.helpers import NAR, SGN, TAR, Heteroskedastic


def run_single_ts(y: pd.Series, repetition: int) -> pd.DataFrame:
    n = len(y)
    split = int(n * 0.7)
    train, test = y[:split].values.reshape((-1, 1)), y[split:].values.reshape((-1, 1))

    pred_deepARMA = get_deep_arma(train, test, n, y.name, repetition)
    pred_deepARMA = pd.Series(pred_deepARMA[0], name=pred_deepARMA.name)

    pred_ShallowARMA = get_shallow_arma(train, test, n, y.name, repetition)
    pred_ShallowARMA = pd.Series(pred_ShallowARMA[0], name=pred_ShallowARMA.name)

    pred_shallow_lstm = get_lstm(train, test, n, y.name, repetition, deep=False)
    pred_shallow_lstm = pd.Series(pred_shallow_lstm[0], name=pred_shallow_lstm.name)

    pred_deep_lstm = get_lstm(train, test, n, y.name, repetition, deep=True)
    pred_deep_lstm = pd.Series(pred_deep_lstm[0], name=pred_deep_lstm.name)

    pred_shallow_gru = get_gru(train, test, n, y.name, repetition, deep=False)
    pred_shallow_gru = pd.Series(pred_shallow_gru[0], name=pred_shallow_gru.name)

    pred_deep_gru = get_gru(train, test, n, y.name, repetition, deep=True)
    pred_deep_gru = pd.Series(pred_deep_gru[0], name=pred_deep_gru.name)

    pred_shallow_simple = get_simple(train, test, n, y.name, repetition, deep=False)
    pred_shallow_simple = pd.Series(pred_shallow_simple[0], name=pred_shallow_simple.name)

    pred_deep_simple = get_simple(train, test, n, y.name, repetition, deep=True)
    pred_deep_simple = pd.Series(pred_deep_simple[0], name=pred_deep_simple.name)

    pred_arma = get_autoarima(train.flatten(), test.flatten())

    preds = pd.DataFrame(
        [pred_arma, pred_deepARMA, pred_ShallowARMA, pred_shallow_lstm, pred_deep_lstm, pred_shallow_gru, pred_deep_gru,
         pred_shallow_simple, pred_deep_simple, pd.Series(test, name=y.name)]).T

    return preds


def single_rep(repetition: int, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_all_seeds(repetition)
    arparams = np.array([0.1, 0.3])
    maparams = np.array([-0.4])
    alpha_true = 0
    y1 = pd.Series(simulate_arma_process(arparams, maparams, alpha_true, n_steps=n, std=2), name="ARMA")
    y2 = pd.Series(TAR(n), name="TAR")
    y3 = pd.Series(SGN(n), name="SGN")
    y4 = pd.Series(NAR(n), name="NAR")
    y5 = pd.Series(Heteroskedastic(n), name="Heteroskedastic")
    ys = pd.DataFrame([y1, y2, y3, y4, y5]).T

    single_ts = [ys[col] for col in ys.columns]

    rmse = []
    mae = []
    for single_t in single_ts:
        predictions = run_single_ts(single_t, repetition)
        rmse_metrics = pd.Series(
            {predictions.columns[i]: get_rmse(predictions.iloc[:, i], predictions.iloc[:, -1])
             for i in range(len(predictions.columns) - 1)}, name=single_t.name
        ).to_frame()
        rmse.append(rmse_metrics)
        mae_metrics = pd.Series(
            {predictions.columns[i]: get_mae(predictions.iloc[:, i], predictions.iloc[:, -1])
             for i in range(len(predictions.columns) - 1)}, name=single_t.name
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
    df_rmse.to_csv("simulations/results/univariate_simulation_rmse.csv")
    df_mae = pd.concat([r[1] for r in res], axis=0)
    df_mae.to_csv("simulations/results/univariate_simulation_mae.csv")


if __name__ == "__main__":
    # Usage:
    # python simulations/univariate_simulation.py
    fire.Fire(run)
