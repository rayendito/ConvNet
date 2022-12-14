from src.layers.ConvLayer import ConvLayer
import numpy as np

from src.layers.PoolingLayer import PoolingLayer

# INIT VARIABLES
image = [[[4], [1], [3], [5], [3]], [[2], [1], [1], [2], [2]], [[5], [5], [1], [2], [3]], [[2], [2], [4], [3], [2]], [[5], [1], [3], [4], [5]]]

x_train = np.array([image])

w1 = [[[1, 2, 3], [4, 7, 5], [3, -32, 25]], [[12, 18, 12], [18, -74, 45], [-92, 45, -18]]]
b1 = [0, 0]

w2 = [[1, 2], [3, -4]]
b2 = [0, 0]

W2 = np.array([[0, 1, 2], [0, 3, -4]])

w3 = [[0.09, 0.02], [0.08, 0.03], [0.07, 0.03], [0.06, 0.02], [0.05, 0.01], [0.04, 0.02], [0.03, 0.07], [0.04, 0.08], [0.05, 0.05], [0.01, 0.01]]
b3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

dE_dx3 = np.array([[0, 0.079854, 0]])

# SETUP LAYERS
conv_1 = ConvLayer(2, 3, (5, 5, 1))
pool_2 = PoolingLayer(3, "MAX")

conv_1.kernel = np.array(w1, dtype=float)
conv_1.bias = np.array(b1, dtype=float)

# FORWARD PROP
conv_1_out = conv_1.calculate(x_train)
pool_2_out = pool_2.calculate(conv_1_out)

# BACKWARD PROP
pool_2.update_weights(lr=10e-4, momentum=0, actual=None, preceding_error_term=dE_dx3, preceding_weights=W2, preceding_layer_type="Flatten")
conv_1.update_weights(lr=10e-4, momentum=0, actual=None, preceding_error_term=pool_2.error_term, preceding_weights=None, preceding_layer_type="Pooling")

print(conv_1.kernel)