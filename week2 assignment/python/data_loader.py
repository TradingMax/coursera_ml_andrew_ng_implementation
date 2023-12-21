from pathlib import Path
import numpy as np

path = Path(__file__).parent.parent / 'octave' / 'ex1data1.txt'


def load_data() -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x, y
