# run_transformer_midgamma.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers


# === CONFIG ===
SEQ_LEN = 10
# TARGET_COLUMN = 'mid_gamma'
TARGET_COLUMN = ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta', 'low_gamma', 'mid_gamma']
TEST_SPLIT = 0.2
EPOCHS = 30
BATCH_SIZE = 32
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
#         X.append(data[i:i+seq_len])  # shape = (seq_len, features)
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


def build_transformer(input_shape):
    inputs = tf.keras.Input(shape=input_shape)  # [seq_len, features]

    # Positional encoding via Dense
    x = layers.Dense(64)(inputs)

    # Multi-head attention
    x = layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f" Transformer Evaluation:")
    print(f"  MSE: {mse:.5f}, MAE: {mae:.5f}")

    plt.figure(figsize=(10, 4))
    plt.plot(y_test, label='True')
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title("Transformer Prediction - Mid Gamma")
    plt.xlabel("Time steps (s)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_multichannel(y_true, y_pred, model_name="Model"):
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    print(f"\n {model_name} Multichannel Evaluation:")
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
#     model = build_transformer(input_shape=(X_train.shape[1], X_train.shape[2]))
#
#     model.fit(
#         X_train, y_train,
#         validation_data=(X_test, y_test),
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         verbose=0  # 静音
#     )
#     model.save("saved_models/transformer_model.h5")
#
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#
#     if return_metrics:
#         return mse, mae
#     else:
#         evaluate(model, X_test, y_test)
def main(return_metrics=False):
    data, feature_names, _ = load_and_preprocess(CSV_PATH)
    X, y = create_sequences(data, seq_len=SEQ_LEN)

    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(8)(x)  # 输出8个通道

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    model.save("saved_models/transformer_model.h5")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    if return_metrics:
        return mse, mae
    else:
        evaluate_multichannel(y_test, y_pred, model_name="Transformer")



if __name__ == "__main__":
    main()
