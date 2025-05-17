#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("t_eegdata.csv")
df = df.drop(['secs'],axis=1)
df.dropna(inplace=True)

# 数据归一化
scalerX = MinMaxScaler(feature_range=(0, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))
scaledX = scalerX.fit_transform(df.drop('attention', axis=1))
scaledY = scalerY.fit_transform(df[['attention']])

df_scaled = pd.concat([pd.DataFrame(scaledY, columns=['attention']), pd.DataFrame(scaledX, columns=df.columns[1:])], axis=1) #列方向

# 滑动窗口大小
look_back = 10

# 创建数据集
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 1:]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# 划分训练集和测试集
train_size = int(len(df_scaled) * 0.7)
test_size = len(df_scaled) - train_size
train, test = df_scaled.values[0:train_size,:], df_scaled.values[train_size:len(df_scaled),:]

# 生成训练集和测试集
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 调整输入数据的维度以适应RNN的输入
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# 构建RNN模型
model = Sequential()
model.add(SimpleRNN(4, input_shape=(look_back, 6)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam') #均值误差损失函数 #adam优化器
model.summary()
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
model.save("eeg_attention_rnn_model.h5")  # 保存模型

# 进行预测
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 反归一化
trainPredict = scalerY.inverse_transform(trainPredict)
trainY = scalerY.inverse_transform([trainY])
testPredict = scalerY.inverse_transform(testPredict)
testY = scalerY.inverse_transform([testY])

print('Train Predictions: ', trainPredict)


plt.plot(list(testPredict), label='RNN_predict')
plt.plot(list(testY[0]), label='True')
plt.title('EEG Attention predict')
plt.xlabel('data')
plt.ylabel('shares')
plt.legend()
plt.show()








