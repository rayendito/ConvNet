from src.utils.dataloader import load_cats_and_dogs
from src.layers.ConvLayer import ConvLayer
from src.layers.DenseLayer import DenseLayer
from src.layers.FlattenLayer import FlattenLayer
from src.layers.PoolingLayer import PoolingLayer
from src.model import Model
import numpy as np

dataset = load_cats_and_dogs()

catsndogs = np.array(dataset["data"], dtype=object)
labels = np.array(dataset["labels"])

model = Model()
model.addLayer(ConvLayer(5, 3, (100, 100, 3), stride=1, padding=0))
model.addLayer(ConvLayer(2, 5, (98, 98, 5), stride=1, padding=0))
model.addLayer(PoolingLayer(2, "MAX"))
model.addLayer(FlattenLayer())
model.addLayer(DenseLayer(2048, "relu", batch_size=40, input_size=7500))
model.addLayer(DenseLayer(128, "sigmoid", batch_size=40, input_size=2048))
model.addLayer(DenseLayer(16, "relu", batch_size=40, input_size=128))
model.addLayer(DenseLayer(1, "sigmoid", batch_size=40, input_size=16))

outputs = model.predict(catsndogs)

print(outputs)