from src.model import Model
from src.layers.PoolingLayer import PoolingLayer
from src.utils.utils import save


channels_ = [[[i+j, i+j+11, i+j+22] for i in range(1, 7)] for j in range(6)]
dummy_data_ = [channels_ for _ in range(5)]

# for c in range(3):
#     for e1 in dummy_data_[0]:
#         for e2 in e1:
#             print(e2[c], end="\t")
#         print()

model = Model()
model.addLayer(PoolingLayer(2, "MAX"))
model.addLayer(PoolingLayer(3, "AVG"))

outputs = model.forwardProp(dummy_data_)

for channel in outputs[0]:
    for row in channel:
        print(row)

save(model, "./bin/pooling_model.pkl")
