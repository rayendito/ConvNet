import numpy as np


class FlattenLayer:
    def calculate(self, inputs):
        self.inputs = inputs
        self.outputs = self._flatten(self.inputs)

        return self.outputs
    
    def _flatten(self, inputs):
        return np.array([np.array(datapoint).flatten() for datapoint in inputs])