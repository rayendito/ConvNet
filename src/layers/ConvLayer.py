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

    # MILESTONE B
    def update_weights(self, lr, preceding_error_term, preceding_weights):
        # assume convolutional layer is never output layer

        # dE/dw = dE/dX_i+1 * dX_i+1/dX_Pool *     dX_Pool/dRelu   * dRelu/dX_i * dX_i/dw
        # dE/dw = dE/dX_i+1 *      w_i+1     *  klo max -> neuron  * dRelu/dX_i *  X_i-1
        #                                       max = 1, else 0 | 
        #                                       klo avg -> 1/(n*n)

        if(self.outputs.any() == None):
            raise ValueError('layer has no output, run forward propagation first')
        
        if(preceding_error_term is None or preceding_weights is None):
            raise ValueError('hidden layer weight update requires preceding error terms and preceding weights')
        error_term = self._calculate_error_term_conv(preceding_error_term, preceding_weights)

        for idx, inp in enumerate(self.inputs):
            err_term_on_that_input = error_term[idx]
            weight_updates = []
            for element in inp:
                weight_update = lr*element*err_term_on_that_input
                weight_updates.append(weight_update)
            self.weights += np.transpose(weight_updates)

    # OUTPUT LAYER ERROR TERM

    def _calculate_error_term_output(self, actual):
        output_function_derivative = self._sigmoid_output_function_derivative() if self.activation == 'sigmoid' else self._relu_output_function_derivative()
        error_function_derivative = self._error_function_derivative(actual)
        self.error_term = -1*output_function_derivative*error_function_derivative
        return self.error_term

    def _error_function_derivative(self, actual):
        error_mean = np.array(np.array(actual)-self.outputs).mean(0)
        return -1 * error_mean

    # HIDDEN LAYER ERROR TERM

    def _calculate_error_term_conv(self, preceding_error_term, preceding_weights):
        output_function_derivative = self._sigmoid_output_function_derivative() if self.activation == 'sigmoid' else self._relu_output_function_derivative()
        sum_expression = self._calculate_sum_expression(preceding_error_term, preceding_weights)
        self.error_term = -1*output_function_derivative*sum_expression
        return self.error_term

    def _calculate_sum_expression(self, preceding_error_term, preceding_weights):
        preceding_weights = np.transpose(preceding_weights)
        sum_expressions = [[np.dot(err, weight) for weight in preceding_weights] for err in preceding_error_term]
        return -1*np.array(sum_expressions)
    
    # ACTIVATION FUNCTION DERIVATIVE

    def _sigmoid_output_function_derivative(self):
        return self.outputs*(1-self.outputs)

    def _relu_output_function_derivative(self):
        return np.array([np.vectorize(self._relu_derivative)(outp) for outp in self.outputs])
    
    def _relu_derivative(val):
        return 0 if val < 0 else 1

    # GETTER
    def get_output(self):
        return self.outputs

    def get_error_terms(self):
        return self.error_term

    def get_weights(self):
        return self.weights
