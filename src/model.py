import numpy as np

class Model:
    def __init__(self):
        self.layers = []
        self.lr = None
        self.momentum = None

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
        self.preds = np.round(self.outputs)

        if (len(self.preds[0]) == 1):
            self.preds = self.preds.flatten()
        
        return self.preds

    @staticmethod
    def checkShape(x, layer):
        if ("input_shape" in dir(layer)):
            return x.shape == layer.input_shape
        elif ("batch_size" in dir(layer)):
            return len(x) == layer.batch_size
        else:
            return True

    def compile_model(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum

    def fit(self, inputs, labels, batch_size=4, epoch=5):
        for i in range(epoch):
            for j in range(0,len(inputs), batch_size):
                begin = j
                end = max(j+batch_size, len(inputs))
                self.forwardProp(inputs[begin:end])
                for idx, layer in enumerate(reversed(self.layers)):
                    layer.update_weights(lr=self.lr,
                                            momentum = self.momentum,
                                            actual=labels[begin:end],
                                            preceding_error_term=self.layers[idx-1].error_term if idx > 0 else None,
                                            preceding_weights=self.layers[idx-1].weights if idx > 0 else None)


    def backProp(self, inputs):
        # TODO: Implement mini batch stochastic gradient descent backpropagation
        pass
