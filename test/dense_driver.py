from src.layers.DenseLayer import DenseLayer
import numpy as np

lr = 0.02
inputs = np.array([
    [4,4,4,4,4],
    [6,6,6,6,6],
    [8,8,8,8,8],
])
actual = [
    [1,0,0],
    [0,1,0],
    [0,1,0],
]

hidden_layer = DenseLayer(output_size=5, batch_size=3, input_size=5, activation='sigmoid', is_output_layer=False)
output_layer = DenseLayer(output_size=3, batch_size=3, input_size=5, activation='sigmoid')

# FORWARD PROP
hidden_output = hidden_layer.calculate(inputs)
output_output = output_layer.calculate(hidden_output)

print('HIDDEN LAYER RESULTS')
print(hidden_output)
print('OUTPUT LAYER RESULTS')
print(output_output)

# BACK PROP
print('OUTPUT LAYER WEIGHT AFTER UPDATE WEIGHT')
output_layer.update_weights(lr, actual=actual)
print('OUTPUT LAYER ERROR TERMS')
print(output_layer.get_error_terms())

preced_weights = output_layer.get_weights()
preced_err_terms = output_layer.get_error_terms()

print(preced_weights.shape)
print(preced_err_terms.shape)

print('HIDDEN WEIGHTS BEFORE')
print(hidden_layer.weights)

print('HIDDEN BIASES BEFORE')
print(hidden_layer.biases)

hidden_layer.update_weights(lr, preceding_error_term=preced_err_terms, preceding_weights=preced_weights)

print('HIDDEN WEIGHTS AFTER')
print(hidden_layer.weights)

print('HIDDEN BIASES AFTER')
print(hidden_layer.biases)