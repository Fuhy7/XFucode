# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 读取CSV文件
data = pd.read_csv('t_eegdata.csv')

# 提取输入和输出
X = data['secs'].values
y = data['attention'].values


# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X.reshape(-1, 1), y)

# 预测结果
y_pred = model.predict(X.reshape(-1, 1))

# 绘制结果图
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title('EEG Attention predict')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()
plt.savefig('res.jpg')
