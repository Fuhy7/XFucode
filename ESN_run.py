# fixed_esn_run.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from reservoirpy.nodes import Reservoir, Ridge

import joblib
import os


# === CONFIG ===
SEQ_LEN = 10
TARGET_COLUMN = 'mid_gamma'
TEST_SPLIT = 0.2
CSV_PATH = 'data/eeg_brainwaves_1hour.csv'
################


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    feature_names = df.columns.tolist()
    return scaled, feature_names, scaler


# def create_sequences(data, target_idx, seq_len):
#     X, y = [], []
#     for i in range(len(data) - seq_len):
#         X.append(data[i:i+seq_len].flatten())  # Flatten (10 × 8 = 80 dims)
#         y.append(data[i+seq_len][target_idx])
#     return np.array(X), np.array(y)
def create_sequences(data, seq_len=10, flatten=False):
    X, y = [], []
    for i in range(len(data) - seq_len):
        seq_x = data[i:i+seq_len]
        target_y = data[i+seq_len]
        X.append(seq_x if not flatten else seq_x.flatten())
        y.append(target_y)
    return np.array(X), np.array(y)


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"📊 ESN Evaluation:")
    print(f"  MSE: {mse:.5f}, MAE: {mae:.5f}")

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title("ESN Prediction - Mid Gamma")
    plt.xlabel("Time steps (s)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_multichannel(y_true, y_pred, model_name="Model"):
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    print(f"\n{model_name} Multichannel Evaluation:")
    for i, name in enumerate(['delta', 'theta', 'low_alpha', 'high_alpha',
                              'low_beta', 'high_beta', 'low_gamma', 'mid_gamma']):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        print(f"  {name:<12} - MSE: {mse:.5f}")
        # 可选：画图
        plt.figure(figsize=(8, 3))
        plt.plot(y_true[:, i], label='True', linewidth=1.2)
        plt.plot(y_pred[:, i], label='Predicted', linestyle='--')
        plt.title(f"{model_name} - {name} Prediction")
        plt.xlabel("Time steps")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

# def main(return_metrics=False):
#     data, feature_names, _ = load_and_preprocess(CSV_PATH)
#     target_idx = feature_names.index(TARGET_COLUMN)
#
#     X, y = create_sequences(data, target_idx, SEQ_LEN)
#     split_idx = int(len(X) * (1 - TEST_SPLIT))
#     X_train, X_test = X[:split_idx], X[split_idx:]
#     y_train, y_test = y[:split_idx], y[split_idx:]
#
#     y_train = y_train.reshape(-1, 1)
#     y_test = y_test.reshape(-1)
#
#     reservoir = Reservoir(units=200, sr=0.9, input_scaling=0.5)
#     readout = Ridge(ridge=1e-6)
#
#     states_train = reservoir.run(X_train)
#     readout.fit(states_train, y_train)
#
#     os.makedirs("saved_models", exist_ok=True)
#
#     # 保存模型对象
#     joblib.dump(reservoir, "saved_models/esn_reservoir.pkl")
#     joblib.dump(readout, "saved_models/esn_readout.pkl")
#
#     y_pred = readout.run(reservoir.run(X_test)).reshape(-1)
#
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#
#     if return_metrics:
#         return mse, mae
#     else:
#         evaluate(y_test, y_pred)
def main(return_metrics=False):
    data, feature_names, _ = load_and_preprocess(CSV_PATH)
    X, y = create_sequences(data, seq_len=SEQ_LEN, flatten=True)  # X: [n,80]；y: [n,8]

    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    y_train = y_train.reshape(-1, 8)
    y_test = y_test.reshape(-1, 8)

    reservoir = Reservoir(units=200, sr=0.9, input_scaling=0.5)
    readout = Ridge(ridge=1e-6)

    states_train = reservoir.run(X_train)
    readout.fit(states_train, y_train)

    os.makedirs("saved_models", exist_ok=True)

    # 保存模型对象
    joblib.dump(reservoir, "saved_models/esn_reservoir.pkl")
    joblib.dump(readout, "saved_models/esn_readout.pkl")

    y_pred = readout.run(reservoir.run(X_test))

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    if return_metrics:
        return mse, mae
    else:
        evaluate_multichannel(y_test, y_pred, model_name="ESN")


if __name__ == "__main__":
    main()
