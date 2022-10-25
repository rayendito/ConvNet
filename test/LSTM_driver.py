from src.layers import LSTMLayer
import numpy as np

test_input = np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3],
])

elestiem = LSTMLayer(5, input_size=3)
elestiem.calculate(test_input)