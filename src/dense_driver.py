from layers.DenseLayer import DenseLayer
import numpy as np

dense = DenseLayer(batch_size=3, input_size=5, output_size=4, activation='sigmoid')


inputs = np.array([
    [1,1,1,1,1],
    [2,2,2,2,2],
    [3,3,3,3,3],
])

print(dense.calculate(inputs=inputs))