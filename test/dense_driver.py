from src.layers.DenseLayer import DenseLayer
import numpy as np

dense = DenseLayer(output_size=5, activation='relu', testing=True)


inputs = np.array([
    [4,4,4,4,4],
    [6,6,6,6,6],
    [4,4,4,4,4],
    [6,6,6,6,6],
    [4,4,4,4,4],
    [6,6,6,6,6],
])

print(dense.calculate(inputs=inputs))