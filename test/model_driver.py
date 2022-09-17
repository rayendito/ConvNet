from src.utils.dataloader import load_cats_and_dogs
from src.layers.ConvLayer import ConvLayer
from src.layers.DenseLayer import DenseLayer
from src.layers.FlattenLayer import FlattenLayer
from src.layers.PoolingLayer import PoolingLayer
from src.model import Model
import numpy as np


catsndogs = load_cats_and_dogs()["data"]
for i in range(len(catsndogs)):
    print(np.array(catsndogs[i], dtype=object).shape)
model = Model()
# model.addLayer(ConvLayer())