import numpy as np

class Model:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forwardProp(self, inputs):
        self.inputs = inputs
        x = np.array(self.inputs)

        for layer in self.layers:
            if Model.checkShape(x, layer):
                x = layer.calculate(x)

        return x

    def predict(self, inputs):
        self.outputs = self.forwardProp(inputs)

        return np.round(self.outputs)

    @staticmethod
    def checkShape(x, layer):
        if ("input_shape" in dir(layer)):
            return x.shape == layer.input_shape
        elif ("batch_size" in dir(layer)):
            return len(x) == layer.batch_size
        else:
            return True


    def backProp(self, inputs):
        # TODO: Implement mini batch stochastic gradient descent backpropagation

        pass
