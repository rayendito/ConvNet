from platform import release
import numpy as np
from src.utils.utils import sigmoid, ReLU

class DenseLayer:
    # MILESTONE A

    def __init__(self, output_size, activation, batch_size=10, input_size=10, testing=False):
        # type checking
        if (not isinstance(batch_size, int)):
            raise TypeError('DenseLayer batch_size must be an integer')
        elif (not isinstance(input_size, int)):
            raise TypeError('DenseLayer input_size must be an integer')
        elif (not isinstance(output_size, int)):
            raise TypeError('DenseLayer output_size must be an integer')
        elif (not isinstance(activation, str)):
            raise TypeError('DenseLayer activation must be a string')

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.testing = testing
        
        if(activation == 'sigmoid'):
            self.activation = sigmoid
        elif(activation == 'relu'):
            self.activation = ReLU
        else:
            raise ValueError('Unknown activation function'+self.activation)
        
        self.weights = self._test_initialize_weights(input_size, output_size) if (testing) else self._initialize_weights(input_size, output_size)
        self.biases = self._test_initialize_biases(output_size) if (testing) else self._initialize_biases(output_size)
        self.inputs = None
        self.net = None
        self.outputs = None
    
    def calculate(self, inputs):
        self._resize_batch_and_input_size_if_necessary(inputs)
        self.inputs = np.array(inputs)
        
        # check batch size
        if (len(inputs) != self.batch_size):
            raise IndexError('Expected batch size to be {}, got {}'.format(self.batch_size, len(self.inputs)))

        inputs_times_weights = self._calc_inputs_v_weights()
        nets = self._add_biases(inputs_times_weights)
        self.outputs = self._run_activation_function(nets)
        return self.outputs

    def _resize_batch_and_input_size_if_necessary(self, inputs):
        inp_batch_size, inp_input_size = np.array(inputs, dtype=object).shape
        if((inp_batch_size, inp_input_size) != (self.batch_size, self.input_size)):
            self.batch_size = inp_batch_size
            self.input_size = inp_input_size
            self.weights = self._test_initialize_weights(inp_input_size, self.output_size) if (self.testing) else self._initialize_weights(inp_input_size, self.output_size)
            self.biases = self._test_initialize_biases(self.output_size) if (self.testing) else self._initialize_biases(self.output_size)

    def _initialize_weights(self, input_size, output_size):
        return np.random.uniform(low=-1, high=1, size=(output_size,input_size))
    
    def _test_initialize_weights(self, input_size, output_size):
        return np.array([[0.1]*input_size]*output_size)

    def _initialize_biases(self, output_size):
        return np.random.uniform(low=-1, high=1, size=output_size)

    def _test_initialize_biases(self, output_size):
        return np.array([0.2]*output_size)

    def _calc_inputs_v_weights(self):
        return np.transpose(np.matmul(self.weights, np.transpose(self.inputs)))
    
    def _add_biases(self, inputs_times_weights):
        return np.array([input_times_weight + self.biases for input_times_weight in inputs_times_weights])

    def _run_activation_function(self, nets):
        return np.vectorize(self.activation)(nets)

    # MILESTONE B
    def update_weights(self, lr, actual):
        if(self.outputs.any() == None):
            raise ValueError('Layer has no output, run forward propagation first')
        error_term = self._error_term(actual)

        for idx, inp in enumerate(self.inputs):
            err_term_on_that_input = error_term[idx]

            weight_updates = []
            for element in inp:
                weight_update = lr*element*err_term_on_that_input
                weight_updates.append(weight_update)

            self.weights += np.transpose(weight_updates)

        #TODO: delete wleowleowleo
        print(self.weights)

    def _error_term(self, actual):
        error_function_derivative = self._error_function_derivative(actual)
        output_function_derivative = self._sigmoid_output_function_derivative() if self.activation == 'sigmoid' else self._relu_output_function_derivative()
        return -1*error_function_derivative*output_function_derivative

    def _error_function_derivative(self, actual):
        return -1 * np.array(np.array(actual)-self.outputs)

    def _sigmoid_output_function_derivative(self):
        return self.outputs*(1-self.outputs)

    def _relu_output_function_derivative(self):
        #TODO: change to actually implement the derivative of the relu function :D
        return self.outputs*(1-self.outputs)


# my star, my perfect silence
# ===================
#        ____
#        |  |
#      __|__|__
#      ( ' > ') ? 
#      /   _  \
#     /:  / \ ;\ 
#      \  \_/ /
#       vv--vv
# ===================