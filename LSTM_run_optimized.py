
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

SEQ_LEN = 10
TEST_SPLIT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
CSV_PATH = 'data/eeg_brainwaves_1hour.csv'

def load_and_preprocess(path):
    df = pd.read_csv(path).drop(columns=['timestamp'], errors='ignore')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, df.columns.tolist(), scaler

def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(64),
        Dense(64, activation='gelu'),
        Dense(8)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main(return_metrics=False):
    data, _, _ = load_and_preprocess(CSV_PATH)
    X, y = create_sequences(data, SEQ_LEN)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    model = build_lstm((SEQ_LEN, X.shape[2]))
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/lstm_model.h5")
    if not return_metrics:
        y_pred = model.predict(X_test)
        print("LSTM MSE:", mean_squared_error(y_test, y_pred))

if __name__ == '__main__':
    main()
