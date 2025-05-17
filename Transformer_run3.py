import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# 参数设置
SEQ_LEN = 50  #时间步
TEST_SPLIT = 0.2
EPOCHS = 100
BATCH_SIZE = 32
CSV_PATH = 'data/eeg_brainwaves_1hour.csv'

# 数据预处理
def load_and_preprocess(path):
    df = pd.read_csv(path).drop(columns=['timestamp'], errors='ignore')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, df.columns.tolist(), scaler

#  生成序列
def create_sequences(data, seq_len=50):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# Transformer 模型
def build_transformer(input_shape, num_heads=4, ff_dim=128, num_blocks=2, dropout_rate=0.1):
    seq_len, feature_dim = input_shape
    inputs = Input(shape=input_shape)

    # 可学习位置编码
    pos_indices = tf.range(start=0, limit=seq_len, delta=1)
    pos_embed_layer = layers.Embedding(input_dim=seq_len, output_dim=feature_dim)  # 可训练的位置嵌入层
    pos_encoding = pos_embed_layer(pos_indices)
    x = inputs + pos_encoding

    # 多个Gated AttentionBlock
    for _ in range(num_blocks):
        # 多头注意力层
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)

        gate = layers.Dense(feature_dim, activation='sigmoid')(x)  # shape: (batch, seq_len, features)
        # 门控机制
        gated_output = gate * attn_output + (1 - gate) * x  # 加权残差连接

        x = layers.LayerNormalization()(gated_output)

        # 前馈网络
        ff = layers.Dense(ff_dim, activation='gelu')(x)
        ff = layers.Dense(feature_dim)(ff)
        ff = layers.Dropout(dropout_rate)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)  # 归一化

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='gelu')(x)
    outputs = layers.Dense(feature_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='mse', metrics=['mae'])
    return model

def main(return_metrics=False):
    # 读取与准备数据
    data, _, _ = load_and_preprocess(CSV_PATH)
    X, y = create_sequences(data, SEQ_LEN)
    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # 构建模型
    model = build_transformer((SEQ_LEN, X.shape[2]))

    # 设置回调函数
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, verbose=1)
    ]

    # 训练模型
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=callbacks,
              verbose=1)

    # # 保存模型
    # os.makedirs("saved_models", exist_ok=True)
    # model.save("saved_models/transformer_optimized_model.h5")

    # 测试结果
    if not return_metrics:
        y_pred = model.predict(X_test)
        print("Transformer Optimized MSE:", mean_squared_error(y_test, y_pred))
        # print("Transformer Optimized MAE:", mean_absolute_error(y_test, y_pred))

if __name__ == '__main__':
    main()
