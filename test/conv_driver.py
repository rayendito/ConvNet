from src.layers.ConvLayer import ConvLayer
import numpy as np

layer1 = ConvLayer(1,3,(5,5,3),stride=1,padding=0)
print(layer1.calculate(np.random.rand(3,5,5,3)))