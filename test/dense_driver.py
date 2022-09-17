from src.layers.DenseLayer import DenseLayer
import numpy as np

dense = DenseLayer(output_size=3, activation='sigmoid')

output_size = 3
input_size = 5

print(np.random.uniform(low=-1, high=1, size=(output_size,input_size)))
print(np.random.uniform(low=-1, high=1, size=output_size))

inputs = np.array([
    [4,4,4,4,4],
    [6,6,6,6,6],
    [4,4,4,4,4],
    [6,6,6,6,6],
    [4,4,4,4,4],
    [6,6,6,6,6],
])

print(dense.calculate(inputs=inputs))