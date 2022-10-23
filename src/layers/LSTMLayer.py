import numpy as np
from src.utils.utils import sigmoid, ReLU

class LSTMLayer:
    def __init__(self, num_cells, input_size=10):
        if (not isinstance(num_cells, int)):
            raise TypeError('LSTMLayer num_cells must be an integer')
        
        self.layer_type = 'LSTM'

        self.Uf = self._initialize_matrix_random(num_cells, input_size)
        self.Ui = self._initialize_matrix_random(num_cells, input_size)
        self.Ucdd = self._initialize_matrix_random(num_cells, input_size)
        self.Uo = self._initialize_matrix_random(num_cells, input_size)

        self.Wf = self._initialize_matrix_random(num_cells, num_cells)
        self.Wi = self._initialize_matrix_random(num_cells, num_cells)
        self.Wcdd = self._initialize_matrix_random(num_cells, num_cells)
        self.Wo = self._initialize_matrix_random(num_cells, num_cells)

        self.Bf = self._initialize_matrix_random(num_cells, 1)
        self.Bi = self._initialize_matrix_random(num_cells, 1)
        self.Bcdd = self._initialize_matrix_random(num_cells, 1)
        self.Bo = self._initialize_matrix_random(num_cells, 1)

        self.h_before = self._initialize_array_zeros(num_cells)
        self.C_before = self._initialize_array_zeros(num_cells)
    
        self.Ft = None
        self.it = None
        self.Cddt = None
        self.Ot = None

        self.all_net = None

    def _initialize_matrix_random(self, n_rows, n_columns):
        return np.random.uniform(low=-1, high=1, size=(n_rows, n_columns))

    def _initialize_array_zeros(self, size):
        return np.array([0.0]*size)

    def calculate(self, input_sequence):
        for i in input_sequence:
            pass
        pass

    def _calculate_Ft(self):
        pass

    def _calculate_it(self):
        pass

    def _calculate_Cddt(self):
        pass

    def _calculate_Ot(self):
        pass

    def _calculate_CellState(self):
        pass

    def _calculate_HiddenState(self):
        pass

# my star, my perfect silence
# ===================
#        ____
#        |  |
#      __|__|__
#      ( ' > ') ~
#      /   _  \
#     /:  / \ ;\ 
#      \  \_/ /
#       vv--vv
# ===================