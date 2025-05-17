# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter


# 1. 数据预处理与增强


# 加载数据
df = pd.read_csv("t_eegdata.csv")
df = df.drop(['secs'], axis=1)
df.dropna(inplace=True)

# 平滑滤波 - 使用 Savitzky-Golay 滤波器进行数据去噪
df[df.columns[1:]] = df[df.columns[1:]].apply(lambda x: savgol_filter(x, window_length=5, polyorder=2))

# 数据归一化
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))
scaledX = scalerX.fit_transform(df.drop('attention', axis=1))
scaledY = scalerY.fit_transform(df[['attention']])

df_scaled = pd.concat([pd.DataFrame(scaledY, columns=['attention']), pd.DataFrame(scaledX, columns=df.columns[1:])],
                      axis=1)  # 列方向

# 滑动窗口大小 - 增大 look_back 以捕获更多历史特征
look_back = 20


# 创建数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 1:]  # 特征部分
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # 目标值
    return np.array(dataX), np.array(dataY)


# 划分训练集和测试集
train_size = int(len(df_scaled) * 0.7)
train, test = df_scaled.values[0:train_size, :], df_scaled.values[train_size:len(df_scaled), :]

# 生成训练集和测试集
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 重新调整输入数据的维度
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))


#  2. 数据增强：添加高斯噪声

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


trainX_noisy = add_noise(trainX)
testX_noisy = add_noise(testX)


#  3. Transformer 模型定义


# 位置编码函数
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # sin 作用于偶数索引位置
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # cos 作用于奇数索引位置
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# Transformer Encoder 模块
def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout_rate=0.2):
    # Multi-Head Self-Attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = layers.Add()([inputs, x])

    # 前馈网络
    ffn = layers.Dense(ff_dim, activation="relu")(res)
    ffn = layers.Dense(inputs.shape[-1])(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    ffn_out = layers.LayerNormalization(epsilon=1e-6)(ffn + res)

    return ffn_out


# 构建 Transformer 模型
def build_transformer(input_shape, num_heads=8, key_dim=8, ff_dim=128, num_layers=3):
    inputs = layers.Input(shape=input_shape)

    # 位置编码
    pos_encoding = positional_encoding(input_shape[0], input_shape[1])
    x = inputs + pos_encoding

    # 堆叠多层 Transformer Encoder
    for _ in range(num_layers):
        x = transformer_encoder(x, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout_rate=0.2)

    # 输出层
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)  # 增加 Dropout 防止过拟合
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


#  4. 构建和编译模型


model = build_transformer((look_back, 6))  # 输入形状 (look_back, 6)
model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error')
model.summary()


#  5. 训练模型并引入回调


# EarlyStopping 和 ReduceLROnPlateau 防止过拟合
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# 训练模型
history = model.fit(trainX_noisy, trainY,
                    epochs=100,
                    batch_size=64,  # 增大批量大小
                    validation_data=(testX_noisy, testY),
                    callbacks=[early_stop, reduce_lr],
                    verbose=2)

# 保存模型
model.save("model/eeg_attention_transformer_model_optimized_v2.h5")


#  6. 进行预测


# 进行预测
trainPredict = model.predict(trainX_noisy)
testPredict = model.predict(testX_noisy)

# 反归一化
trainPredict = scalerY.inverse_transform(trainPredict.reshape(-1, 1))
trainY = scalerY.inverse_transform(trainY.reshape(-1, 1))
testPredict = scalerY.inverse_transform(testPredict.reshape(-1, 1))
testY = scalerY.inverse_transform(testY.reshape(-1, 1))


#  7. 可视化结果


# 绘制训练损失和验证损失
plt.plot(history.history['loss'], label='训练集损失')
plt.plot(history.history['val_loss'], label='验证集损失')
plt.title('训练过程 - 损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE 损失')
plt.legend()
plt.grid(True)
plt.show()

# 绘制预测结果
plt.plot(testPredict, label='预测值 (Predicted)', linestyle='--')
plt.plot(testY, label='真实值 (True)', linestyle='-')
plt.title('EEG 注意力预测结果')
plt.xlabel('时间点')
plt.ylabel('注意力值')
plt.legend()
plt.grid(True)
plt.show()
