# Supplementary material for the paper:
# ARMA cell: a modular and effective approach for neural autoregressive modeling

## Note
This repo is intended to facilitate reproducing the results in the paper.
To use the ARMA cell in your own project, install it from the [software repo](https://github.com/phschiele/armacell).

## Getting started

The syntax of the ARMA cell is similar to existing RNN models in tensorflow, with the additional parameter q for the numer of MA lags.
The number of AR lags are already represented in the preprocessed data.

Below is an example using the functional model API
```python
x = ARMA(q, input_dim=(n_features, p), units=1, activation="relu", use_bias=True)(x)
 ```

The syntax for the ConvARMA cell is also similar to existing spatiotemporal RNN models. If "return_lags" is True, a subsequent 
ConvARMA layer will itself have multiple lags.
```python
x = ConvARMA(
    activation="relu",
    units=64,
    q=3,
    image_shape=image_shape,
    kernel_size=(3, 3),
    return_lags=False,
    return_sequences=True
)(x)
```



## Simulations
The simulation results can be recreated by running
```shell
python simulations/univariate_simulation.py
python simulations/multivariate_simulation.py
```
The output contains predictions for each model as well as the true labels, such that performance metrics can be calculated.

To recreate the parameter recovery plot, run
```shell
python simulations/arma_parameter_recovery.py
```

## Benchmarks
The uni- and multivariate benchmarks can be recreated by running
```shell
python benchmarks/m4_script.py
python benchmarks/traffic_script.py
python benchmarks/elec_script.py
python benchmarks/exchange_script.py
```
yielding the predictions for all models.

For the tensorvariate time series, it is recommended to use a GPU backend to achieve a faster runtime. The benchmarks can
be recreated by running
```shell
python benchmarks/moving_mnist.py
python benchmarks/moving_squares.py
python benchmarks/nyc_data.py
```

## Test
Unit and regression tests are handled through `pytest`, which can be installed via `pip install pytest`.
To run all tests, simply execute
```shell
pytest
```
from the root of the repository.


## Dependency management
During development, dependencies were managed via `poetry`. The provided `pyproject.toml` and `poetry.lock` files allow
exact replication of the package versions by running `poetry install`.

As a fallback, we also provide the exact dependency structure in `requirements_full.txt`.
In addition, `requirements_main.txt` only contains a list of the top level dependencies required for running the experiments.

Further, we note that as of writing this documentation, the default package versions installed on Google Colab are compatible with our
code.
