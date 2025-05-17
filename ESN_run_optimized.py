
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from reservoirpy.nodes import Reservoir, Ridge
import joblib
import os

SEQ_LEN = 10
TEST_SPLIT = 0.2
CSV_PATH = 'data/eeg_brainwaves_1hour.csv'

def load_and_preprocess(path):
    df = pd.read_csv(path).drop(columns=['timestamp'], errors='ignore')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, df.columns.tolist(), scaler

def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        seq_x = data[i:i+seq_len].flatten()
        target_y = data[i+seq_len]
        X.append(seq_x)
        y.append(target_y)
    return np.array(X), np.array(y)

def main(return_metrics=False):
    data, _, _ = load_and_preprocess(CSV_PATH)
    X, y = create_sequences(data, SEQ_LEN)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    reservoir = Reservoir(units=300, sr=1.1, input_scaling=0.7)
    readout = Ridge(ridge=1e-5)
    states_train = reservoir.run(X_train)
    readout.fit(states_train, y_train)
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(reservoir, "saved_models/esn_reservoir.pkl")
    joblib.dump(readout, "saved_models/esn_readout.pkl")
    if not return_metrics:
        y_pred = readout.run(reservoir.run(X_test))
        print("ESN MSE:", mean_squared_error(y_test, y_pred))

if __name__ == '__main__':
    main()
