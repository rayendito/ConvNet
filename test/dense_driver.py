from src.layers.DenseLayer import DenseLayer
import numpy as np

output_size = 3
input_size = 5

dense = DenseLayer(output_size=output_size, batch_size=3, input_size=input_size, activation='sigmoid')

inputs = np.array([
    [4,4,4,4,4],
    [6,6,6,6,6],
    [8,8,8,8,8],
])

# MILESTONE B TESTING REGION
lr = 0.2
actual = [
    [1,0,0],
    [0,1,0],
    [0,1,0],
]
print('WEIGHTS BEFORE')
print(dense.weights)

print('RESULTS')
print(dense.calculate(inputs=inputs))

print('WEIGHTS AFTER')
dense.update_weights(lr, actual)