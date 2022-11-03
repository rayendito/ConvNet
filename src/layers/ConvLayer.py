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
        self.layer_type = "Convolution"
        self.last_update_weights = np.zeros((n_filters, kernel_size, kernel_size))
        self.outputShape = (None, input_shape[0], input_shape[1], n_filters)
        self.param = self.n_filters * self.kernel_size * self.kernel_size * self.input_shape[0] + self.n_filters

    
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
        # image = np.pad(image, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')

        # Get shapes
        H, W, C = self.input_shape
        # Get image shape
        Hi, Wi, Ci = image.shape
        if( Hi != H or Wi != W or Ci != C):
            raise ValueError('dimension mismatch expected of size {}, got {}'.format((H, W, C), (Hi, Wi, Ci)))
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
                        output[h, w, f] += np.sum(image[h * self.stride:h * self.stride + kH, w * self.stride:w * self.stride + kW, c] * self.kernel[f])
                    output[h, w, f] += self.bias[f]
        
        return output
    
    # def calculate_output_size(self):
    #     return ((self.input_shape[0] - self.kernel_size + 2*self.padding) // self.stride + 1,(self.input_shape[1] - self.kernel_size + 2*self.padding) // self.stride + 1, self.input_shape[2])
    
    def calculate(self,input):
        self.input = input
        self.output = [[] for _ in self.input]
        for a in range(len(self.input)):
            self.input[a] = np.pad(self.input[a], ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant')
            result = self.convolution(self.input[a])
            self.output[a] = result
        self.output = np.array(self._run_activation_function(self.output))

        return self.output
    
    def _run_activation_function(self, image):
        return np.vectorize(self.activation)(image)

    # MILESTONE B
    def update_weights(self, lr=10e-4, momentum = 0, actual=None, preceding_error_term=None, preceding_weights=None, preceding_layer_type=None):
        # assume convolutional layer is never output layer

        # dE/dw = dE/dX_i+1 * dX_i+1/dX_Pool *     dX_Pool/dRelu   * dRelu/dX_i * dX_i/dw
        # dE/dw = dE/dX_i+1 *      w_i+1     *  klo max -> neuron  * dRelu/dX_i *  X_i-1
        #                                       max = 1, else 0 | 
        #                                       klo avg -> 1/(n*n)

        if(len(self.output) == 0):
            raise ValueError('layer has no output, run forward propagation first')
        
        if(preceding_error_term is None):
            raise ValueError('hidden layer weight update requires preceding error terms and preceding weights')
        error_term = self._calculate_error_term_conv(preceding_error_term, preceding_weights, preceding_layer_type)

        for idx, inp in enumerate(self.input):
            err_term_on_that_input = error_term[idx]
            conv_derivative = self._convolution_derivative(inp)

            weight_updates = np.zeros(self.kernel.shape)

            for i, nest_1 in enumerate(conv_derivative):
                for j, nest_2 in enumerate(nest_1):
                    for k, nest_3 in enumerate(nest_2):
                        np.add(weight_updates[k], -1*lr*nest_3*err_term_on_that_input[i][j][k], out=weight_updates[k], casting="unsafe")

            self.kernel += np.array(weight_updates, dtype=float)

            self.bias += np.array([np.sum(err_term_on_that_input[i]) for i in range(self.n_filters)])

    # CONVOLUTION LAYER ERROR TERM

    def _calculate_error_term_conv(self, preceding_error_term, preceding_weights, preceding_layer_type):
        output_function_derivative = self._relu_output_function_derivative()

        if (preceding_layer_type == "Flatten"):
            intermediate_term = preceding_error_term*(preceding_error_term*preceding_weights)
            self.error_term = intermediate_term*output_function_derivative
        elif (preceding_layer_type == "Pooling"):
            self.error_term = preceding_error_term*output_function_derivative
        elif (preceding_layer_type == "Convolution"):
            # intermediate_term = preceding_error_term*np.array([self._kernel_derivative(self.output[0]) for _ in self.output], dtype=object)
            # self.error_term = -1*intermediate_term*output_function_derivative
            self.error_term = preceding_error_term*output_function_derivative
        return self.error_term
    
    # ACTIVATION FUNCTION DERIVATIVE

    def _relu_output_function_derivative(self):
        return 1*(self.output > 0)

    # CONVOLUTION LAYER DERIVATIVE
    def _convolution_derivative(self, image):
        H, W, C = self.input_shape
        # Get image shape
        Hi, Wi, Ci = image.shape
        if( Hi != H or Wi != W or Ci != C):
            raise ValueError('dimension mismatch expected of size {}, got {}'.format((H, W, C), (Hi, Wi, Ci)))
        F = self.n_filters
        kH, kW = self.kernel_size, self.kernel_size
        # Compute output shape
        H_ = int((H - kH) / self.stride + 1)
        W_ = int((W - kW) / self.stride + 1)
        # Initialize output
        output = [[[[[0 for _ in range(kH)] for _ in range(kW)] for _ in range(F)] for _ in range(W_)] for _ in range(H_)]
        # Convolution
        for h in range(H_):
            for w in range(W_):
                for f in range(F):
                    for c in range(C):
                        output[h][w][f] += image[h * self.stride:h * self.stride + kH, w * self.stride:w * self.stride + kW, c]

        return np.array(output, dtype=object)

    # GETTER
    def get_output(self):
        return self.output

    def get_error_terms(self):
        return self.error_term

    def get_weights(self):
        return self.weights

    def get_all_weights(self):
        return self.weights
