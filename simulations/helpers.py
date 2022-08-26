import numpy as np


def TAR(n: int) -> np.ndarray:
    y = np.zeros(n)
    for i in range(1, n):
        if np.abs(y[i]) <= 1:
            y[i] = 0.9 * y[i - 1] + np.random.randn()
        else:
            y[i] = -0.3 * y[i - 1] + np.random.randn()
    return y


def SGN(n: int) -> np.ndarray:
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = np.sign(y[i - 1]) + np.random.randn()
    return y


def NAR(n: int) -> np.ndarray:
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = (0.7 * np.abs(y[i - 1])) / (np.abs(y[i - 1]) + 2) + np.random.randn()
    return y


def Heteroskedastic(n: int) -> np.ndarray:
    y = np.zeros(n)
    eps = np.random.randn(n)
    for i in range(n):
        if i == 0:
            y[i] = eps[i]
        elif i == 1:
            y[i] = eps[i] - 0.4 * eps[i - 1]
        else:
            y[i] = eps[i] - 0.4 * eps[i - 1] + 0.3 * eps[i - 2] + 0.5 * eps[i - 1] * eps[i - 2]
    return y


def SQ(n: int) -> np.ndarray:
    x, y = np.zeros(n), np.zeros(n)
    eps, e = np.random.randn(n), np.random.randn(n)
    for i in range(n):
        if i == 0:
            x[i] = e[i]
            y[i] = eps[i]
        else:
            x[i] = 0.6 * x[i - 1] + e[i]
            y[i] = x[i] ** 2 + eps[i]
    return np.stack([x, y], axis=-1)


def EXP(n: int) -> np.ndarray:
    x, y = np.zeros(n), np.zeros(n)
    eps, e = np.random.randn(n), np.random.randn(n)
    for i in range(n):
        if i == 0:
            x[i] = e[i]
            y[i] = eps[i]
        else:
            x[i] = 0.6 * x[i - 1] + e[i]
            y[i] = np.exp(x[i]) + eps[i]
    return np.stack([x, y], axis=-1)
