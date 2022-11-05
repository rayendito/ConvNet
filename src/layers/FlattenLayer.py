import numpy as np


class FlattenLayer:
    def __init__(self):
        self.layer_type = "Flatten"
        self.param = 0

    def calculate(self, inputs):
        self.inputs = inputs
        self.outputs = self._flatten(self.inputs)

        return self.outputs
    
    def _flatten(self, inputs):
        return np.array([np.array(datapoint).flatten() for datapoint in inputs])

    def update_weights(self, lr=10e-4, momentum = 0, actual=None, preceding_error_term=None, preceding_weights=None, preceding_layer_type=None):
        self.error_term = preceding_error_term
        self.weights = preceding_weights

    def get_all_weights(self):
        return self.weights