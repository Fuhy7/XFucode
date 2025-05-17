# #!/usr/bin/env python
# # coding: utf-8
#
# import pandas as pd
# import numpy as np
# from keras.models import Sequential, Model
# from keras.layers import Input, Dense, LSTM, TimeDistributed
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
#
# # 加载数据
# df = pd.read_csv("t_eegdata.csv")
#
# # 删除不需要的列
# df = df.drop(['secs'], axis=1)
# df.dropna(inplace=True)
#
# # 数据归一化
# scalerX = MinMaxScaler(feature_range=(0, 1))
# scalerY = MinMaxScaler(feature_range=(0, 1))
#
# # 特征数据（去掉 'attention' 列，剩余的列都是特征）
# scaledX = scalerX.fit_transform(df.drop('attention', axis=1))
#
# # 标签数据（只保留 'attention' 列）
# scaledY = scalerY.fit_transform(df[['attention']])
#
# df_scaled = pd.concat([pd.DataFrame(scaledY, columns=['attention']), pd.DataFrame(scaledX, columns=df.columns[1:])], axis=1)
#
# # 滑动窗口大小
# look_back = 10
#
# # 创建数据集
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 1:]  # 选择特征部分（不包括attention）
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])  # 选择目标列（attention）
#     return np.array(dataX), np.array(dataY)
#
# # 划分训练集和测试集
# train_size = int(len(df_scaled) * 0.7)
# test_size = len(df_scaled) - train_size
# train, test = df_scaled.values[0:train_size, :], df_scaled.values[train_size:len(df_scaled), :]
#
# # 生成训练集和测试集
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
#
# # 调整输入数据的维度以适应LSTM的输入
# trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
# testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
#
# # 查看输入数据的形状
# print(f"trainX.shape: {trainX.shape}")
# print(f"testX.shape: {testX.shape}")
#
# # 创建自编码器模型
# def build_autoencoder(input_shape):
#     print(f"Building autoencoder with input_shape: {input_shape}")
#     input_layer = Input(shape=input_shape)
#     encoded = TimeDistributed(Dense(8, activation='relu'))(input_layer)  # 压缩到一个低维表示，每个时间步应用相同的编码器
#     decoded = TimeDistributed(Dense(input_shape[1], activation='sigmoid'))(encoded)  # 解码回原始特征维度
#     autoencoder = Model(input_layer, decoded)
#     encoder = Model(input_layer, encoded)  # 提取编码部分
#     autoencoder.compile(optimizer='adam', loss='mean_squared_error')
#     return autoencoder, encoder
#
# # 自编码器的输入维度
# input_shape = (trainX.shape[1], trainX.shape[2])  # (时间步数, 特征数量)
#
# # 构建自编码器模型
# autoencoder, encoder = build_autoencoder(input_shape)
#
# # 训练自编码器
# autoencoder.fit(trainX, trainX, epochs=50, batch_size=32, verbose=2)
#
# # 获取自编码器的编码部分输出（压缩后的特征）
# train_encoded = encoder.predict(trainX)
# test_encoded = encoder.predict(testX)
#
# # 构建LSTM模型
# model = Sequential()
# model.add(LSTM(50, input_shape=(train_encoded.shape[1], train_encoded.shape[2])))  # 使用压缩后的特征作为LSTM的输入
# model.add(Dense(1))  # 输出预测值
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # 训练LSTM模型
# model.fit(train_encoded, trainY, epochs=100, batch_size=1, verbose=2)
#
# # # 加载旧模型（.h5 格式）
# # model = load_model("model/lstm_eeg_model.h5", compile=False)
#
# # 重新保存为 SavedModel 格式
# model.save("model/lstm_eeg_model_converted", save_format="tf")
#
#
#
# # 进行预测
# trainPredict = model.predict(train_encoded)
# testPredict = model.predict(test_encoded)
#
# # 反归一化
# trainPredict = scalerY.inverse_transform(trainPredict)
# trainY = scalerY.inverse_transform([trainY])
# testPredict = scalerY.inverse_transform(testPredict)
# testY = scalerY.inverse_transform([testY])
#
# # 打印预测结果
# print('Train Predictions:', trainPredict)
#
# # 绘制预测结果
# plt.plot(list(testPredict), label='LSTM Predict')
# plt.plot(list(testY[0]), label='True')
# plt.title('EEG Attention Prediction')
# plt.xlabel('Data')
# plt.ylabel('Attention Level')
# plt.legend()
# plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('t_eegdata.csv')  # 请替换为你的文件名

# 只选择需要的特征列
features = ['attention', 'mytheta', 'mydelta', 'mylow_alpha', 'myhigh_alpha', 'mylow_beta', 'myhigh_beta']
data = data[features]

# 归一化处理
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 创建前10秒的数据来预测下一秒
sequence_length = 10
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])  # 前10秒
    y.append(scaled_data[i])                    # 第11秒

X = np.array(X)
y = np.array(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dense(X.shape[2]))
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# 预测
predicted = model.predict(X_test)

# 反归一化
predicted_inverse = scaler.inverse_transform(predicted)
y_test_inverse = scaler.inverse_transform(y_test)

# 可视化结果（比如 attention）
plt.figure(figsize=(10, 4))
plt.plot(y_test_inverse[:, 0], label='True Attention')
plt.plot(predicted_inverse[:, 0], label='Predicted Attention')
plt.legend()
plt.title('Attention Prediction (10s Input)')
plt.xlabel('Time Step')
plt.ylabel('Attention')
plt.tight_layout()
plt.show()
