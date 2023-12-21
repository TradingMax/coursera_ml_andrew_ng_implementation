from pathlib import Path
import numpy as np

path_data1 = Path(__file__).parent.parent / 'octave' / 'ex1data1.txt'
path_data2 = Path(__file__).parent.parent / 'octave' / 'ex1data2.txt'


def load_data1() -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path_data1, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def load_data2() -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path_data2, delimiter=',')
    x = data[:, :2]
    y = data[:, 2]
    return x, y
