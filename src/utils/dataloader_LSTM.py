import pandas as pd
import numpy as np


def load_LSTM(window_size=10):

    df = pd.read_csv("./data_LSTM/ETH-USD-Test.csv")
    df2 = df.tail(32)   

    open = df2["Open"].to_numpy()
    open_min = np.min(open)
    open_max = np.max(open)
    open = (open - open_min) / (open_max - open_min)
    open = np.array([open[i:i+window_size] for i in range(len(open)-window_size+1)])
    
    high = df2["High"].to_numpy()
    high_min = np.min(high)
    high_max = np.max(high)
    high = (high - high_min) / (high_max - high_min)
    high = np.array([high[i:i+window_size] for i in range(len(high)-window_size+1)])

    low = df2["Low"].to_numpy()
    low_min = np.min(low)
    low_max = np.max(low)
    low = (low - low_min) / (low_max - low_min)
    low = np.array([low[i:i+window_size] for i in range(len(low)-window_size+1)])
    
    close = df2["Close"].to_numpy()
    close_min = np.min(close)
    close_max = np.max(close)
    close = (close - close_min) / (close_max - close_min)
    close = np.array([close[i:i+window_size] for i in range(len(close)-window_size+1)])

    volume = df2["Volume"].to_numpy()
    volume_min = np.min(volume)
    volume_max = np.max(volume)
    volume = (volume - volume_min) / (volume_max - volume_min)
    volume = np.array([volume[i:i+window_size] for i in range(len(volume)-window_size+1)])
    

    dict = {
        "open" : open,
        "open_min" : open_min,
        "open_max" : open_max,
        "high" : high,
        "high_min" : high_min,
        "high_max" : high_max,
        "low" : low,
        "low_min" : low_min,
        "low_max" : low_max,
        "close" : close,
        "close_min" : close_min,
        "close_max" : close_max,
        "volume" : volume,
        "volume_min" : volume_min,
        "volume_max" : volume_max
    }

    return dict