import numpy as np


def hypothesis(x: float, w1: float, w2: float) -> float:
    return w1 + w2 * x


def err(x: float, y: float, w1: float, w2: float) -> float:
    return hypothesis(x, w1, w2) - y


def mse(x: float, y: float, w1: float, w2: float) -> float:
    return err(x, y, w1, w2) ** 2


def cost_function(x_arr: np.ndarray, y_arr: np.ndarray, w1: float, w2: float) -> float:
    running_sum = 0
    for x, y in zip(x_arr, y_arr):
        running_sum += mse(x, y, w1, w2)

    return running_sum / (2 * len(x_arr))


def derivative_cost_function_w1(x_arr: np.ndarray, y_arr: np.ndarray, w1: float, w2: float):
    running_sum = 0
    for x, y in zip(x_arr, y_arr):
        running_sum += err(x, y, w1, w2)
    return running_sum / len(x_arr)


def derivative_cost_function_w2(x_arr: np.ndarray, y_arr: np.ndarray, w1: float, w2: float):
    running_sum = 0
    for x, y in zip(x_arr, y_arr):
        running_sum += err(x, y, w1, w2) * x
    return running_sum / len(x_arr)


def run_gradient_descent(x_arr: np.ndarray, y_arr: np.ndarray, w1_init: float, w2_init: float, a: float):
    tolerance = 1e-16
    max_iter = 1500
    w1, w2 = w1_init, w2_init
    d1 = d2 = 1
    i = 0
    while abs(d1) > tolerance and abs(d2) > tolerance and i < max_iter:
        d1 = derivative_cost_function_w1(x_arr, y_arr, w1, w2)
        d2 = derivative_cost_function_w2(x_arr, y_arr, w1, w2)
        w1 = w1 - a * d1
        w2 = w2 - a * d2
        i += 1

    return w1, w2
