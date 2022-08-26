from test.test_regression_arma_cell import get_trained_ARMA_p_q_model

import numpy as np
from arma_cell.helpers import prepare_arma_input, simulate_arma_process, set_all_seeds
from arma_cell.plotting import plot_convergence
from statsmodels.tsa.arima.model import ARIMA


def main():

    set_all_seeds()
    arparams = np.array([0.1, 0.3])
    maparams = np.array([-0.4])
    alpha_true = 0

    p = len(arparams)
    q = len(maparams)
    add_intercept = False

    y_arma = simulate_arma_process(arparams, maparams, alpha_true, n_steps=25000, std=2)

    arima_model = ARIMA(endog=y_arma, order=(p, 0, q), trend="c" if add_intercept else "n").fit()  # order = (p,d,q)

    X_train, y_train = prepare_arma_input(max(p, q), y_arma)
    tf_model = get_trained_ARMA_p_q_model(q, X_train, y_train, add_intercept, plot_training=True)

    plot_convergence(tf_model, p, add_intercept, arima_model, "simulations/figures/arma_parameter_recovery.pdf")


if __name__ == '__main__':
    main()
