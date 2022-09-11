class Model:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forwardProp(self, inputs):
        self.inputs = inputs
        x = self.inputs

        for layer in self.layers:
            x = layer.calculate(x)

        return x

    def backProp(self, inputs):
        # TODO: Implement mini batch stochastic gradient descent backpropagation

        pass
