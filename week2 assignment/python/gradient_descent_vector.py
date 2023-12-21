import numpy as np
import warnings


def hypothesis(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.dot(x, w)


def err(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return hypothesis(x, w) - y


def mse(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return err(x, y, w) ** 2


def cost_function(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    err_vec = err(x, y, w)
    mse_sum = np.dot(err_vec, err_vec.T)
    return mse_sum / (2 * len(x))


def derivative_cost_function(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.dot(x.T, err(x, y, w)) / len(x)


def run_gradient_descent(x: np.ndarray,
                         y: np.ndarray,
                         w_init: np.ndarray,
                         a: float,
                         max_iter: int = 1500,
                         tolerance: float = 1e-16) -> np.ndarray:
    w = w_init
    d = np.ones((len(w),))
    i = 0
    while all(abs(d) > tolerance) and i < max_iter:
        d = derivative_cost_function(x, y, w)
        w = w - a * d
        i += 1

    return w


if __name__ == '__main__':
    from data_loader import load_data
    x_test, y_test = load_data()
    x_test = np.vstack([np.ones(len(x_test)), x_test]).T  # Add a row of ones to x_test
    w_test = run_gradient_descent(x_test, y_test, np.array([0.1, 0.1]), 1)

    print()
