from src.layers.LSTMLayer import LSTMLayer
from src.utils.dataloader_LSTM import load_LSTM
from src.layers.DenseLayer import DenseLayer
from src.model import Model
import numpy as np

data_LSTM = load_LSTM(window_size=4)

model3 = Model()
model3.addLayer(LSTMLayer(10, input_size=4))
model3.addLayer(DenseLayer(1))

model3.summary()

outputs = model3.predict(np.array(data_LSTM["open"]))
print("open prediction: ", outputs * (data_LSTM["open_max"] - data_LSTM["open_min"]) + data_LSTM["open_min"])

outputs = model3.predict(np.array(data_LSTM["high"]))
print("high prediction: ", outputs * (data_LSTM["high_max"] - data_LSTM["high_min"]) + data_LSTM["high_min"])

outputs = model3.predict(np.array(data_LSTM["low"]))
print("low prediction: ", outputs * (data_LSTM["low_max"] - data_LSTM["low_min"]) + data_LSTM["low_min"])

outputs = model3.predict(np.array(data_LSTM["close"]))
print("close prediction: ", outputs * (data_LSTM["close_max"] - data_LSTM["close_min"]) + data_LSTM["close_min"])

outputs = model3.predict(np.array(data_LSTM["volume"]))
print("volume prediction: ", outputs * (data_LSTM["volume_max"] - data_LSTM["volume_min"]) + data_LSTM["volume_min"])