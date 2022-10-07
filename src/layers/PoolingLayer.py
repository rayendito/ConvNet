import numpy as np


class PoolingLayer:
    MAX = "MAX"
    AVG = "AVG"

    def __init__(self, size, mode, stride=None):
        # check input types
        if (not isinstance(size, int)):
            raise TypeError('PoolingLayer size must be integer!')
        elif (not isinstance(mode, str)):
            raise TypeError('PoolingLayer mode must be string!')
        elif (stride is not None and not isinstance(stride, int)):
            raise TypeError('PoolingLayer stride must be integer!')

        self.size = size

        # verify input values
        if (mode.upper() == PoolingLayer.MAX or mode.upper() == PoolingLayer.AVG):
            self.mode = mode

            # set pooling function
            if (self.mode == PoolingLayer.MAX):
                self.poolingFunction = PoolingLayer.maxPool
            else:
                assert(self.mode == PoolingLayer.AVG)
                self.poolingFunction = PoolingLayer.averagePool
        else:
            raise ValueError('Pooling mode must be "MAX" or "AVG"')

        if (stride is None):
            self.stride = size
        else:
            self.stride = stride
        
        self.layer_type = "Pooling"

    def calculate(self, inputs):
        # public method calculate
        self.inputs = inputs
        self.outputs = self._pool(self.inputs)

        return self.outputs

    def _pool(self, inputs):
        # POOLING PIPELINE

        # prepare empty output matrix
        outputs = [[[[0 for _ in range(len(channels[0][0]))] for _ in range(0, len(channels[0]) - len(channels[0]) % self.size, self.stride)] for _ in range(0, len(channels) - len(channels) % self.size, self.stride)] for channels in inputs]

        # outputs = [[[[0 for _ in range(0, len(channel[0]) - len(channel[0]) % self.size, self.stride)] for _ in range(
        #     0, len(channel) - len(channel) % self.size, self.stride)] for channel in channels] for channels in inputs]

        # iterate over inputs
        for cs, channels in enumerate(inputs):
            for c in range(len(channels[0][0])):
                for i in range(0, len(channels), self.stride):
                    # iterate through columns with stride
                    if (i + self.size <= len(channels)):
                        for j in range(0, len(channels[i]), self.stride):
                            # iterate through rows with stride
                            if (j + self.size <= len(channels[i])):
                                
                                # prepare matrix of current receptive field
                                currMatrix = [
                                    [0 for _ in range(self.size)] for _ in range(self.size)]

                                # copy current receptive field
                                for a in range(self.size):
                                    for b in range(self.size):
                                        currMatrix[a][b] = channels[i+a][j+b][c]

                                # calculate pooling result
                                outputs[cs][i//self.stride][j // self.stride][c] = self.poolingFunction(currMatrix)

        return outputs
    
    def update_weights(self, lr=10e-4, momentum = 0, actual=None, preceding_error_term=None, preceding_weights=None, _preceding_layer_type=None):
        if (self.mode == PoolingLayer.MAX):
            self.derivatives = [[[[0 for _ in range(len(channels[0][0]))] for _ in range(0, len(channels[0]))] for _ in range(0, len(channels))] for channels in self.inputs]
            
            for cs, channels in enumerate(self.inputs):
                for c in range(len(channels[0][0])):
                    for i in range(0, len(channels), self.stride):
                        # iterate through columns with stride
                        if (i + self.size <= len(channels)):
                            for j in range(0, len(channels[i]), self.stride):
                                # iterate through rows with stride
                                if (j + self.size <= len(channels[i])):
                                    
                                    # prepare matrix of current receptive field
                                    currMatrix = [
                                        [0 for _ in range(self.size)] for _ in range(self.size)]

                                    # copy current receptive field
                                    for a in range(self.size):
                                        for b in range(self.size):
                                            currMatrix[a][b] = channels[i+a][j+b][c]

                                    # calculate pooling result
                                    max_i, max_j = np.unravel_index(currMatrix.argmax(), currMatrix.shape)
                                    self.derivatives[max_i, max_j] = 1
        elif (self.mode == PoolingLayer.AVG):
            self.derivatives = [[[[1/(self.size*self.size) for _ in range(len(channels[0][0]))] for _ in range(0, len(channels[0]))] for _ in range(0, len(channels))] for channels in self.inputs]
        
        self.error_term = (preceding_weights*preceding_error_term)*self.derivatives
        self.weights = None
    
    def get_all_weights(self):
        return self.weights

    @staticmethod
    def maxPool(matrixSlice):
        # max of all elements in slice

        return max(np.array(matrixSlice).flatten())

    @staticmethod
    def averagePool(matrixSlice):
        # average of all elements in slice

        flatMatrix = np.array(matrixSlice).flatten()
        return sum(flatMatrix)/len(flatMatrix)
