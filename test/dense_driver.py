from src.layers.DenseLayer import DenseLayer
import numpy as np

dense = DenseLayer(batch_size=2, input_size=5, output_size=3, activation='relu')


inputs = np.array([
    [4,4,4,4,4],
    [6,6,6,6,6],
])

print(dense.calculate(inputs=inputs))