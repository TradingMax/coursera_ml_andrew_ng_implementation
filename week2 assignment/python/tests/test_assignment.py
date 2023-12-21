import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from data_loader import load_data
import gradient_descent_loop as gdl
import gradient_descent_vector as gdv


@pytest.fixture
def data():
    return load_data()


@pytest.fixture
def x_arr(data):
    return data[0]


@pytest.fixture
def x_arr_ones(x_arr):
    return np.vstack([np.ones(len(x_arr)), x_arr]).T


@pytest.fixture
def y_arr(data) -> np.ndarray:
    return data[1]


@pytest.fixture(params=[[0, 0], [-1, 2]])
def w_init(request):
    return np.array(request.param)


# Updated test functions using the w_init fixture
def test_hypothesis_identical(w_init, x_arr, x_arr_ones):
    hypothesis_loop = np.array([gdl.hypothesis(x, *w_init) for x in x_arr])
    assert_array_almost_equal(hypothesis_loop, gdv.hypothesis(x_arr_ones, w_init), decimal=13)


def test_err_identical(w_init, x_arr, y_arr, x_arr_ones):
    err_loop = np.array([gdl.err(x, y, *w_init) for x, y in zip(x_arr, y_arr)])
    assert_array_almost_equal(err_loop, gdv.err(x_arr_ones, y_arr, w_init), decimal=13)


def test_mse_identical(w_init, x_arr, y_arr, x_arr_ones):
    mse_loop = np.array([gdl.mse(x, y, *w_init) for x, y in zip(x_arr, y_arr)])
    assert_array_almost_equal(mse_loop, gdv.mse(x_arr_ones, y_arr, w_init), decimal=13)


def test_cost_function_identical(w_init, x_arr, y_arr, x_arr_ones):
    assert_array_almost_equal(gdl.cost_function(x_arr, y_arr, *w_init), gdv.cost_function(x_arr_ones, y_arr, w_init),
                              decimal=13)


def test_derivative_cost_function_identical(w_init, x_arr, y_arr, x_arr_ones):
    d1 = gdl.derivative_cost_function_w1(x_arr, y_arr, *w_init)
    d2 = gdl.derivative_cost_function_w2(x_arr, y_arr, *w_init)
    d_vec = gdv.derivative_cost_function(x_arr_ones, y_arr, w_init)
    assert_array_almost_equal(np.array([d1, d2]), d_vec, decimal=13)


def test_run_gradient_descent_identical(w_init, x_arr, y_arr, x_arr_ones):
    w1, w2 = gdl.run_gradient_descent(x_arr, y_arr, *w_init, 0.01)
    w_vec = gdv.run_gradient_descent(x_arr_ones, y_arr, w_init, 0.01)
    assert_array_almost_equal(np.array([w1, w2]), w_vec, decimal=12)


@pytest.mark.parametrize('w1, w2, expected', [
    (0, 0, 32.07),
    (-1, 2, 54.24),
])
def test_cost_function_loop(x_arr, y_arr, w1, w2, expected):
    assert abs(gdl.cost_function(x_arr, y_arr, w1, w2) - expected) < 0.01


@pytest.mark.parametrize('w, expected', [
    (np.array([0, 0]), 32.07),
    (np.array([-1, 2]), 54.24),
])
def test_cost_function_vector(x_arr_ones, y_arr, w, expected):
    assert abs(gdv.cost_function(x_arr_ones, y_arr, w) - expected) < 0.01
