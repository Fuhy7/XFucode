import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import os

SEQ_LEN = 50
TEST_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
CSV_PATH = 'data/eeg_brainwaves_1hour.csv'

# 数据预处理
def load_and_preprocess(path):
    df = pd.read_csv(path).drop(columns=['timestamp'], errors='ignore')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, df.columns.tolist(), scaler

# 创建时间序列数据
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# 构建自编码器模型
def build_autoencoder(input_dim, latent_dim=16):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(latent_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

#  编码整个序列
def encode_sequence_data(X, encoder):
    # 将编码器应用于序列中的每个时间步
    encoded_X = []
    for sample in X:
        encoded_steps = encoder.predict(sample)
        encoded_X.append(encoded_steps)
    return np.array(encoded_X)

# 构建 LSTM 网络模型
def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dense(64, activation='gelu'),
        Dense(input_shape[-1])  # output same as original feature count
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main(return_metrics=False):
    data, feature_names, scaler = load_and_preprocess(CSV_PATH)
    # 构造滑动窗口序列
    X, y = create_sequences(data, SEQ_LEN)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    input_dim = X.shape[2]
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim=input_dim)  # 构造自编码器，形状不改变
    step_data = X_train.reshape(-1, input_dim)  # 展平时间步长来训练自编码器
    autoencoder.fit(step_data, step_data, epochs=30, batch_size=64, verbose=0)

    X_train_encoded = encode_sequence_data(X_train, encoder)
    X_test_encoded = encode_sequence_data(X_test, encoder)

    model = build_lstm((SEQ_LEN, input_dim))
    es = EarlyStopping(patience=7, restore_best_weights=True)

    model.fit(X_train_encoded, y_train, validation_data=(X_test_encoded, y_test),
              epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)

    # os.makedirs("saved_models", exist_ok=True)
    # model.save("saved_models/lstm_ae_model.h5")

    if not return_metrics:
        y_pred = model.predict(X_test_encoded)
        print("AE+LSTM MSE:", mean_squared_error(y_test, y_pred))

if __name__ == '__main__':
    main()
