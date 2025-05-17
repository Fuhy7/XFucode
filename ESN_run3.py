import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from reservoirpy.nodes import Reservoir, Ridge
import joblib
import os
import itertools

SEQ_LEN = 50
TEST_SPLIT = 0.2
CSV_PATH = 'data/eeg_brainwaves_1hour.csv'

#  数据预处理
def load_and_preprocess(path):
    df = pd.read_csv(path).drop(columns=['timestamp'], errors='ignore')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, df.columns.tolist(), scaler

# 构建时间序列数据
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        seq_x = data[i:i+seq_len].flatten()
        target_y = data[i+seq_len]
        X.append(seq_x)
        y.append(target_y)
    return np.array(X), np.array(y)

# ESN 模型评估函数
def evaluate_esn(X_train, y_train, X_test, y_test, units, sr, input_scaling, ridge):
    reservoir = Reservoir(units=units, sr=sr, input_scaling=input_scaling)
    readout = Ridge(ridge=ridge)
    states_train = reservoir.run(X_train)
    readout.fit(states_train, y_train)
    y_pred = readout.run(reservoir.run(X_test))
    mse = mean_squared_error(y_test, y_pred)
    return mse, reservoir, readout

# 网格搜索超参数
def hyperparameter_search(X_train, y_train, X_test, y_test):
    param_grid = {
        "units": [100, 300, 500],
        "sr": [0.9, 1.0, 1.1],
        "input_scaling": [0.5, 0.7, 0.9],
        "ridge": [1e-6, 1e-5, 1e-4]
    }

    best_mse = float("inf")
    best_params = None
    best_reservoir, best_readout = None, None

    for units, sr, input_scaling, ridge in itertools.product(
        param_grid["units"],
        param_grid["sr"],
        param_grid["input_scaling"],
        param_grid["ridge"]
    ):
        mse, reservoir, readout = evaluate_esn(X_train, y_train, X_test, y_test, units, sr, input_scaling, ridge)
        print(f"Params: units={units}, sr={sr}, input_scaling={input_scaling}, ridge={ridge} -> MSE: {mse:.6f}")
        if mse < best_mse:
            best_mse = mse
            best_params = (units, sr, input_scaling, ridge)
            best_reservoir, best_readout = reservoir, readout

    print("\nBest Parameters:", best_params)
    print("Best MSE:", best_mse)

    return best_reservoir, best_readout

def main():
    data, _, _ = load_and_preprocess(CSV_PATH)
    X, y = create_sequences(data, SEQ_LEN)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    best_reservoir, best_readout = hyperparameter_search(X_train, y_train, X_test, y_test)

    # os.makedirs("saved_models", exist_ok=True)
    # joblib.dump(best_reservoir, "saved_models/esn_best_reservoir.pkl")
    # joblib.dump(best_readout, "saved_models/esn_best_readout.pkl")

if __name__ == '__main__':
    main()
