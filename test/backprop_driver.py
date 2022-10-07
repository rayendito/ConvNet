from src.utils.dataloader import load_cats_and_dogs
from src.layers.ConvLayer import ConvLayer
from src.layers.DenseLayer import DenseLayer
from src.layers.FlattenLayer import FlattenLayer
from src.layers.PoolingLayer import PoolingLayer
from src.model import Model
import numpy as np

# INIT VARIABLES
image = [[[4], [1], [3], [5], [3]], [[2], [1], [1], [2], [2]], [[5], [5], [1], [2], [3]], [[2], [2], [4], [3], [2]], [[5], [1], [3], [4], [5]]]

x_train = np.array([image])

w1 = [[[1, 2, 3], [4, 7, 5], [3, -32, 25]], [[12, 18, 12], [18, -74, 45], [-92, 45, -18]]]
b1 = [0, 0]

w2 = [[1, 2], [3, -4]]
b2 = [0, 0]

W2 = np.insert(w2, 0, b2, axis=1)

w3 = [[0.09, 0.02], [0.08, 0.03], [0.07, 0.03], [0.06, 0.02], [0.05, 0.01], [0.04, 0.02], [0.03, 0.07], [0.04, 0.08], [0.05, 0.05], [0.01, 0.01]]
b3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

W3 = np.insert(w3, 0, b3, axis=1)

output = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

# CREATE MODEL
model = Model()
model.addLayer(ConvLayer(2, 3, (5, 5, 1), stride=1, padding=0))
model.addLayer(PoolingLayer(3, "MAX"))
model.addLayer(FlattenLayer())
model.addLayer(DenseLayer(2, "sigmoid", batch_size=1, input_size=1))
model.addLayer(DenseLayer(10, "sigmoid", batch_size=1, input_size=1, is_output_layer=True))

model.layers[0].kernel = np.array(w1, dtype=float)
model.layers[0].bias = np.array(b1, dtype=float)
model.layers[3].weights = np.array(w2, dtype=float)
model.layers[3].bias = np.array(b2, dtype=float)
model.layers[4].weights = np.array(w3, dtype=float)
model.layers[4].bias = np.array(b3, dtype=float)


model.compile_model(10e-1, 0)
print(model.layers[0].kernel)

model.fit(x_train, output, 1, 1)

a = model.predict(x_train)

# model.fit(x_train, output, 1, 100)

# a = model.predict(x_train)
# print(model.layers[0].kernel)