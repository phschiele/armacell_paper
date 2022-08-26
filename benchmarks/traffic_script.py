import os

import fire
import pandas as pd
from pmdarima import utils

from benchmarks.common import run_repeated_benchmark


def run(path: str = "benchmarks/", repetitions: int = 1, compute: bool = True, mode: str = "multivariate",
        metric: str = "rmse", multiprocessing: bool = True) -> None:
    file_path = os.path.join(path, "data/traffic_processed.csv")
    ts_name = "traffic"

    df = pd.read_csv(file_path, index_col=0).dropna(how="all", axis=0).iloc[:, :10]
    df = pd.DataFrame(utils.diff(utils.diff(df.values, lag=24)), columns=df.columns)

    res = run_repeated_benchmark(df, compute, path, ts_name, repetitions, mode, metric, multiprocessing)
    output_path = os.path.join(path, f"results/{mode}_{metric}_{ts_name}.csv")
    res.to_csv(output_path)


if __name__ == "__main__":
    fire.Fire(run)
