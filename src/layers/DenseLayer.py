from platform import release
import numpy as np
from src.utils.utils import sigmoid, ReLU

class DenseLayer:
    # MILESTONE A

    def __init__(self, output_size, activation, batch_size=10, input_size=10, testing=False, is_output_layer=True):
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
        self.is_output_layer = is_output_layer
        
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
        self.error_term = None
        self.last_update_weights = np.array([[0.0]*input_size]*output_size)
        self.last_update_biases = np.array([0.0]*output_size)
    
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
    def update_weights(self, lr, momentum = 0, actual=None, preceding_error_term=None, preceding_weights=None):
        if(self.outputs.any() == None):
            raise ValueError('layer has no output, run forward propagation first')

        if(self.is_output_layer):
            if(actual is None):
                raise ValueError('output layer weight update requires actual values of the prediction')
            error_term = self._calculate_error_term_output(actual)
        else:
            if(preceding_error_term is None or preceding_weights is None):
                raise ValueError('hidden layer weight update requires preceding error terms and preceding weights')
            error_term = self._calculate_error_term_hidden(preceding_error_term, preceding_weights)

        sum_update_weight = np.array([[0.0]*self.input_size]*self.output_size)
        sum_update_bias = np.array([0.0]*self.output_size)
        for idx, inp in enumerate(self.inputs):
            err_term_on_that_input = error_term[idx]
            weight_updates = []
            for element in inp:
                weight_update = lr*element*err_term_on_that_input
                weight_updates.append(weight_update)

            w_update = np.transpose(weight_updates) + momentum*self.last_update_weights
            self.weights += w_update
            sum_update_weight += w_update

            b_update = lr*err_term_on_that_input + momentum*self.last_update_biases
            self.biases += b_update
            sum_update_bias += b_update
        
        self.last_update_weights = sum_update_weight
        self.last_update_biases = sum_update_bias


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

    def _calculate_error_term_hidden(self, preceding_error_term, preceding_weights):
        output_function_derivative = self._sigmoid_output_function_derivative() if self.activation == 'sigmoid' else self._relu_output_function_derivative()
        sum_expression = self._calculate_sum_expression(preceding_error_term, preceding_weights)
        self.error_term = -1*output_function_derivative*sum_expression
        return self.error_term

    def _calculate_sum_expression(self, preceding_error_term, preceding_weights):
        sum_expressions = []
        preceding_weights = np.transpose(preceding_weights)
        for err in preceding_error_term:
            sum_expressions.append([np.dot(err, weight) for weight in preceding_weights])
        return -1*np.array(sum_expressions)
    
    # ACTIVATION FUNCTION DERIVATIVE

    def _sigmoid_output_function_derivative(self):
        return self.outputs*(1-self.outputs)

    def _relu_output_function_derivative(self):
        return np.array([np.vectorize(self._relu_derivative)(outp) for outp in self.outputs])
    
    def _relu_derivative(self, val):
        return 0 if val < 0 else 1

    # GETTER
    def get_output(self):
        return self.outputs

    def get_error_terms(self):
        return self.error_term

    def get_weights(self):
        return self.weights

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