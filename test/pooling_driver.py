from src.model import Model
from src.layers.pooling_layer import PoolingLayer
from src.utils.utils import save

channel_ = [[e+i for e in [1, 2, 3, 4, 5, 6]] for i in range(6)]
channels_ = [[[e+i*11 for e in row] for row in channel_] for i in range(3)]
dummy_data_ = [channels_ for i in range(5)]

model = Model()
model.addLayer(PoolingLayer(2, "MAX"))
model.addLayer(PoolingLayer(3, "AVG"))

outputs = model.forwardProp(dummy_data_)

for channel in outputs[0]:
    for row in channel:
        print(row)

save(model, "../bin/pooling_model.pkl")
