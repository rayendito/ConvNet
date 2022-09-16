import numpy as np
from src.utils.utils import ReLU

class ConvLayer:
    def __init__(self,n_filters,kernel_size,input_shape,stride=1,padding=0,):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.kernel = np.random.rand(n_filters,kernel_size,kernel_size)
        self.stride = stride
        self.activation = ReLU
        self.padding = padding
        self.input_shape = input_shape
        # self.output_shape = self.calculate_output_size()
        self.bias = np.random.rand(self.n_filters)
        self.input = []
        self.output = []

    
    def convolution(self, image):
        """
        Convolution of RGB image with kernel.
        Args:
            image: numpy array of shape (H, W, C).
            kernel: numpy array of shape (kH, kW).
            stride: int.
            padding: int.
        Returns:
            numpy array of shape (H', W', C).
        """
        # Padding
        image = np.pad(image, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
        # Get shapes
        H, W, C = self.input_shape
        F = self.n_filters
        kH, kW = self.kernel_size, self.kernel_size
        # Compute output shape
        H_ = int((H - kH) / self.stride + 1)
        W_ = int((W - kW) / self.stride + 1)
        # Initialize output
        output = np.zeros((H_, W_, F))
        # Convolution
        for h in range(H_):
            for w in range(W_):
                for f in range(F):
                    for c in range(C):
                        output[h, w, f] += np.sum(image[h * self.stride:h * self.stride + kH, w * self.stride:w * self.stride + kW, c] * self.kernel)
                    output[h, w, f] += self.bias[f]

        return output
    
    # def calculate_output_size(self):
    #     return ((self.input_shape[0] - self.kernel_size + 2*self.padding) // self.stride + 1,(self.input_shape[1] - self.kernel_size + 2*self.padding) // self.stride + 1, self.input_shape[2])
    
    def calculate(self,input):
        self.input = input
        for a in range(len(self.input)):
            result = self.convolution(self.input[a])
            self.output.append(result)
        self.output = self._run_activation_function(self.output)
        return self.output
    
    def _run_activation_function(self, image):
        return np.vectorize(self.activation)(image)
