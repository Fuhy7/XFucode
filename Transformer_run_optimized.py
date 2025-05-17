
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
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

def build_transformer(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(64)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(8)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def main(return_metrics=False):
    data, _, _ = load_and_preprocess(CSV_PATH)
    X, y = create_sequences(data, SEQ_LEN)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    model = build_transformer((SEQ_LEN, X.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/transformer_model.h5")
    if not return_metrics:
        y_pred = model.predict(X_test)
        print("Transformer MSE:", mean_squared_error(y_test, y_pred))

if __name__ == '__main__':
    main()
