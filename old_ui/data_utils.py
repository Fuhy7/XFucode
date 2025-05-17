import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_eeg_csv(path):
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    return df

def create_sequences(data, seq_len=10, flatten=False):
    """
    多通道版本：返回
    X: [samples, seq_len, features]
    y: [samples, features]（每个样本预测下一时刻的全通道值）
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        seq_x = data[i:i+seq_len]
        target_y = data[i+seq_len]  # 下一时刻的所有通道
        X.append(seq_x if not flatten else seq_x.flatten())
        y.append(target_y)
    return np.array(X), np.array(y)
