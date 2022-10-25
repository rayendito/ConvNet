import numpy as np
from src.utils.utils import sigmoid, ReLU, tanh

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

        self.Bf = self._initialize_array_random(num_cells)
        self.Bi = self._initialize_array_random(num_cells)
        self.Bcdd = self._initialize_array_random(num_cells)
        self.Bo = self._initialize_array_random(num_cells)

        self.h_before = self._initialize_array_zeros(num_cells)
        self.C_before = self._initialize_array_zeros(num_cells)
    
        self.Ft = self._initialize_array_zeros(num_cells)
        self.it = self._initialize_array_zeros(num_cells)
        self.Cddt = self._initialize_array_zeros(num_cells)
        self.Ot = self._initialize_array_zeros(num_cells)

    def _initialize_matrix_random(self, n_rows, n_columns):
        return np.random.uniform(low=-1, high=1, size=(n_rows, n_columns))

    def _initialize_array_random(self, size):
        return np.random.uniform(low=-1, high=1, size=(size))

    def _initialize_array_zeros(self, size):
        return np.array([0.0]*size)

    def calculate(self, input_sequence):
        '''
            input_sequence example (input dimension: 5):
            [
                [1,2,3,4,5], -> timestep 1
                [1,2,3,4,5], -> timestep 2
                [1,2,3,4,5], -> timestep 3
            ]
        '''
        for timestep in input_sequence:
            self._calculate_Ft(timestep)
            # self._calculate_it()
            # self._calculate_Cddt()
            # self._calculate_Ot()

            # self._calculate_CellState()
            # self._calculate_HiddenState()
        pass

    def _calculate_Ft(self, timestep_input):
        netFt = self._calculate_net('forget', timestep_input)
        print("NETFT YGY")
        print(netFt)
        self.Ft = np.vectorize(sigmoid)(netFt)

    def _calculate_it(self):
        pass

    def _calculate_Cddt(self):
        pass

    def _calculate_Ot(self):
        pass

    def _calculate_net(self, what_gate, timestep_input):
        if(what_gate == 'forget'):
            U = self.Uf
            W = self.Wf
            b = self.Bf
        elif(what_gate == 'input'):
            U = self.Ui
            W = self.Wi
            b = self.Bi
        elif(what_gate == 'candidate'):
            U = self.Ucdd
            W = self.Wcdd
            b = self.Bcdd
        elif(what_gate == 'output'):
            U = self.Uo
            W = self.Wo
            b = self.Bo
        else:
            raise TypeError('Unrecognized LSTM cell type')

        Ux = np.array([sum(s) for s in U*timestep_input])
        print("UX YGY")
        print(Ux)
        Wh_before = np.array([sum(s) for s in W*self.h_before])
        print("WHBEFORE YGY")
        print(Wh_before)
        print('B YGY')
        print(b)
        
        return Ux + Wh_before + b

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