from ..utils.utils import flatten


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

    def calculate(self, inputs):
        # public method calculate
        self.inputs = inputs
        self.outputs = self.pool(self.inputs)

        return self.outputs

    def pool(self, inputs):
        # POOLING PIPELINE

        # prepare empty output matrix
        outputs = [[[[0 for _ in range(0, len(channel[0]) - len(channel[0]) % self.size, self.stride)] for _ in range(
            0, len(channel) - len(channel) % self.size, self.stride)] for channel in channels] for channels in inputs]

        # iterate over inputs
        for cs, channels in enumerate(inputs):
            for c, channel in enumerate(channels):
                for i in range(0, len(channel), self.stride):
                    # iterate through columns with stride
                    if (i + self.size <= len(channel)):
                        for j in range(0, len(channel[i]), self.stride):
                            # iterate through rows with stride
                            if (j + self.size <= len(channel[i])):
                                # prepare matrix of current receptive field
                                currMatrix = [
                                    [0 for _ in range(self.size)] for _ in range(self.size)]

                                # copy current receptive field
                                for a in range(self.size):
                                    for b in range(self.size):
                                        currMatrix[a][b] = channel[i+a][j+b]

                                # calculate pooling result
                                outputs[cs][c][i//self.stride][j //
                                                               self.stride] = self.poolingFunction(currMatrix)

        return outputs

    @staticmethod
    def maxPool(matrixSlice):
        # max of all elements in slice

        return max(flatten(matrixSlice))

    @staticmethod
    def averagePool(matrixSlice):
        # average of all elements in slice

        flatMatrix = flatten(matrixSlice)
        return sum(flatMatrix)/len(flatMatrix)
