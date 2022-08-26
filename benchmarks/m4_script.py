import os

import fire
import pandas as pd
from pmdarima import utils

from benchmarks.common import run_repeated_benchmark


def run(path: str = "benchmarks/", repetitions: int = 1, compute: bool = True, mode: str = "multivariate",
        metric: str = "rmse", multiprocessing: bool = True) -> None:
    file_path = os.path.join(path, "data/m4_hourly_processed.csv")
    ts_name = "m4"

    df_raw = pd.read_csv(file_path, index_col=0).drop(columns=["t_ime"])

    first_valid_indices = []
    for col in df_raw.columns:
        s = df_raw[col]
        first_valid_indices.append(s.first_valid_index())

    start_dates = pd.Series(first_valid_indices, index=df_raw.columns)
    common_start = start_dates.value_counts().idxmax()
    sub_df = df_raw.loc[:, start_dates == common_start].dropna()

    n_ts = 10
    df = pd.DataFrame(utils.diff(utils.diff(sub_df.values, lag=24))[:, :n_ts], columns=sub_df.columns[:n_ts])

    res = run_repeated_benchmark(df, compute, path, ts_name, repetitions, mode, metric, multiprocessing)
    output_path = os.path.join(path, f"results/{mode}_{metric}_{ts_name}.csv")
    res.to_csv(output_path)


if __name__ == "__main__":
    fire.Fire(run)
