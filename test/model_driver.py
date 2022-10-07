from src.utils.dataloader import load_cats_and_dogs
from src.layers.ConvLayer import ConvLayer
from src.layers.DenseLayer import DenseLayer
from src.layers.FlattenLayer import FlattenLayer
from src.layers.PoolingLayer import PoolingLayer
from src.model import Model
import numpy as np
from sklearn.metrics import confusion_matrix

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

# outputs = model.predict(catsndogs)

# 90-10 split
x_train, x_test = np.split(catsndogs, [int(.9*len(catsndogs))])
y_train, y_test = np.split(labels, [int(.9*len(labels))])
model.fit(x_train, y_train, batch_size=4, epoch=5)
outputs = model.predict(x_test)
print("90-10 split test result:")   
print(confusion_matrix(np.array(y_test), np.array(outputs)))

# 10 fold cross validation
outputs2 = []
y_test_total = []
for i in range(10):
    model2 = Model()
    model2.addLayer(ConvLayer(5, 3, (100, 100, 3), stride=1, padding=0))
    model2.addLayer(ConvLayer(2, 5, (98, 98, 5), stride=1, padding=0))
    model2.addLayer(PoolingLayer(2, "MAX"))
    model2.addLayer(FlattenLayer())
    model2.addLayer(DenseLayer(2048, "relu", batch_size=40, input_size=7500))
    model2.addLayer(DenseLayer(128, "sigmoid", batch_size=40, input_size=2048))
    model2.addLayer(DenseLayer(16, "relu", batch_size=40, input_size=128))
    model2.addLayer(DenseLayer(1, "sigmoid", batch_size=40, input_size=16))


    x_train2 = np.concatenate((catsndogs[:i*len(catsndogs)//10], catsndogs[(i+1)*len(catsndogs)//10:]))
    y_train2 = np.concatenate((labels[:i*len(labels)//10], labels[(i+1)*len(labels)//10:]))
    x_test2 = catsndogs[i*len(catsndogs)//10:(i+1)*len(catsndogs)//10]
    y_test2 = labels[i*len(labels)//10:(i+1)*len(labels)//10]
    model2.fit(x_train2, y_train2, batch_size=4, epoch=5)
    outputs2 += model.predict(x_test)
    y_test_total += y_test2

print("10 fold cross validation test result:")
print(confusion_matrix(np.array(y_test_total), np.array(outputs2)))



print(outputs)