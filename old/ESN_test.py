import pandas as pd
import numpy as np
from pyESN import ESN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("t_eegdata.csv")
df = df.drop(['secs'], axis=1)
df.dropna(inplace=True)

# 特征工程：添加滚动特征
df['rolling_mean'] = df['attention'].rolling(window=5).mean()
df['rolling_std'] = df['attention'].rolling(window=5).std()
df.fillna(0, inplace=True)

# 数据归一化
scalerX = StandardScaler()
scalerY = StandardScaler()
scaledX = scalerX.fit_transform(df.drop('attention', axis=1))
scaledY = scalerY.fit_transform(df[['attention']])

df_scaled = pd.concat([pd.DataFrame(scaledY, columns=['attention']), pd.DataFrame(scaledX, columns=df.columns[1:])],
                      axis=1)

# 滑动窗口大小
look_back = 10


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 1:]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# 超参数优化
def train_and_evaluate(trainX, trainY, testX, testY):
    best_rmse = float('inf')
    best_model = None

    for n_reservoir in [100, 200, 500]:
        for sparsity in [0.1, 0.2, 0.5]:
            for spectral_radius in [0.8, 0.9, 0.95]:
                print(
                    f"Training ESN with n_reservoir={n_reservoir}, sparsity={sparsity}, spectral_radius={spectral_radius}")
                esn = ESN(
                    n_inputs=trainX.shape[1],
                    n_outputs=1,
                    n_reservoir=n_reservoir,
                    sparsity=sparsity,
                    spectral_radius=spectral_radius,
                    random_state=42
                )
                esn.fit(trainX, trainY)
                testPredict = esn.predict(testX)
                testPredict_inv = scalerY.inverse_transform(testPredict)
                testY_inv = scalerY.inverse_transform([testY])
                rmse = np.sqrt(mean_squared_error(testY_inv[0], testPredict_inv))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = esn

    print(f"Best RMSE: {best_rmse}")
    return best_model


# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(df_scaled):
    train, test = df_scaled.values[train_index], df_scaled.values[test_index]
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = trainX.reshape(trainX.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)

    # 训练模型
    best_model = train_and_evaluate(trainX, trainY, testX, testY)

# 保存最佳模型
joblib.dump(best_model, "model/esn_eeg_model_optimized.pkl")

# 预测
trainPredict = best_model.predict(trainX)
testPredict = best_model.predict(testX)

# 反归一化
trainPredict = scalerY.inverse_transform(trainPredict)
trainY = scalerY.inverse_transform([trainY])
testPredict = scalerY.inverse_transform(testPredict)
testY = scalerY.inverse_transform([testY])

# 计算评估指标
print(f"Train RMSE: {np.sqrt(mean_squared_error(trainY[0], trainPredict))}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(testY[0], testPredict))}")
print(f"Test MAE: {mean_absolute_error(testY[0], testPredict)}")
print(f"Test R2 Score: {r2_score(testY[0], testPredict)}")

# 绘制预测结果
plt.figure()
plt.plot(list(testPredict), label='ESN Predict')
plt.plot(list(testY[0]), label='True')
plt.title('EEG Attention Prediction (Optimized ESN)')
plt.xlabel('Data')
plt.ylabel('Attention Level')
plt.legend()
plt.show()