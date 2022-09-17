from src.model import Model
from src.layers.FlattenLayer import FlattenLayer

channels_ = [[[i+j, i+j+11, i+j+22] for i in range(1, 7)] for j in range(6)]
dummy_data_ = [channels_ for _ in range(5)]

for c in range(3):
    for e1 in dummy_data_[0]:
        for e2 in e1:
            print(e2[c], end="\t")
        print()

model = Model()
model.addLayer(FlattenLayer())

outputs = model.forwardProp(dummy_data_)

print(outputs[0])