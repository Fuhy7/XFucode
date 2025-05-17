# import numpy as np
# from keras.models import load_model
# import tkinter as tk
# from tkinter import messagebox
# import threading
# import time
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.pyplot as plt
# import pyttsx3
#
# # 加载训练好的模型
# model = load_model("model/eeg_attention_transformer_model.h5")
#
# # 初始化语音引擎
# engine = pyttsx3.init()
#
# # 随机生成数据来模拟 EEG 注意力信号（0-100 的范围）
# def generate_fake_eeg_data():
#     return np.random.rand(1, 10, 6)  # 随机生成 (1, 10, 6) 的数据模拟EEG输入
#
# # 实时监测和绘制波形
# def monitor_and_plot():
#     global stop_monitoring, attention_values
#
#     while not stop_monitoring:
#         # 生成模拟 EEG 数据并预测注意力水平
#         fake_data = generate_fake_eeg_data()
#         predicted_attention = model.predict(fake_data)[0][0] * 100  # 恢复到 0-100 范围
#
#         # 更新注意力值
#         attention_var.set(f"当前注意力值: {round(predicted_attention, 2)}")
#
#         # 将注意力值添加到列表并限制列表长度
#         attention_values.append(predicted_attention)
#         if len(attention_values) > 50:  # 显示最近的 50 个数据点
#             attention_values.pop(0)
#
#         # 检查是否低于阈值并发出语音警告
#         if predicted_attention < attention_threshold:
#             alert_with_voice("注意力水平过低！请集中注意力！")
#
#         # 更新波形图
#         update_plot()
#
#         # 每秒更新一次
#         time.sleep(1)
#
# # 语音警告函数
# def alert_with_voice(message):
#     engine.say(message)
#     engine.runAndWait()
#
# # 更新波形图
# def update_plot():
#     ax.clear()
#     ax.plot(attention_values, color='blue')
#     ax.set_ylim([0, 100])
#     ax.set_title("实时注意力波形图")
#     ax.set_xlabel("时间")
#     ax.set_ylabel("注意力水平")
#     canvas.draw()
#
# # 启动监测线程
# def start_monitoring():
#     global stop_monitoring
#     stop_monitoring = False
#     threading.Thread(target=monitor_and_plot).start()
#
# # 停止监测
# def stop_monitoring_action():
#     global stop_monitoring
#     stop_monitoring = True
#
# # 设置低注意力警告阈值
# attention_threshold = 30  # 低于此值触发警告
#
# # 创建主窗口
# root = tk.Tk()
# root.title("EEG 注意力监测系统")
#
# # 显示注意力值的标签
# attention_var = tk.StringVar()
# attention_label = tk.Label(root, textvariable=attention_var, font=("Helvetica", 16))
# attention_label.pack(pady=20)
#
# # 创建波形图
# fig, ax = plt.subplots(figsize=(6, 4))
# attention_values = []
# canvas = FigureCanvasTkAgg(fig, master=root)
# canvas.get_tk_widget().pack()
#
# # 开始按钮
# start_button = tk.Button(root, text="开始监测", command=start_monitoring, font=("Helvetica", 14))
# start_button.pack(pady=10)
#
# # 停止按钮
# stop_button = tk.Button(root, text="停止监测", command=stop_monitoring_action, font=("Helvetica", 14))
# stop_button.pack(pady=10)
#
# # 初始化停止监测标志
# stop_monitoring = False
#
# # 启动 GUI
# root.mainloop()
import numpy as np
import pyttsx3
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


# 加载训练好的模型
model = load_model("model/eeg_attention_transformer_model.h5")

# 初始化语音引擎
engine = pyttsx3.init()

# 随机生成数据来模拟 EEG 注意力信号（0-100 的范围）
def generate_fake_eeg_data():
    return np.random.rand(1, 10, 6)  # 随机生成 (1, 10, 6) 的数据模拟EEG输入

# 实时监测和绘制波形
def monitor_and_plot():
    global stop_monitoring, attention_values

    while not stop_monitoring:
        # 生成模拟 EEG 数据
        fake_data = generate_fake_eeg_data()

        # 检查数据的形状
        print(f"Input shape to model: {fake_data.shape}")  # 调试：检查输入数据形状

        try:
            # 进行预测
            predicted_attention = model.predict(fake_data)
            print(f"Prediction output shape: {predicted_attention.shape}")  # 调试：检查模型输出形状

            # 如果预测输出有多个维度，取第一维度的第一个值
            predicted_attention = predicted_attention[0][0] * 100  # 恢复到 0-100 范围

            # 更新注意力值
            attention_var.set(f"当前注意力值: {round(predicted_attention, 2)}")

            # 将注意力值添加到列表并限制列表长度
            attention_values.append(predicted_attention)
            if len(attention_values) > 50:  # 显示最近的 50 个数据点
                attention_values.pop(0)

            # 检查是否低于阈值并发出语音警告
            if predicted_attention < attention_threshold:
                alert_with_voice("注意力水平过低！请集中注意力！")

            # 更新波形图
            update_plot()

        except Exception as e:
            print(f"Error during prediction: {e}")

        # 每秒更新一次
        time.sleep(1)

# 语音警告函数
def alert_with_voice(message):
    engine.say(message)
    engine.runAndWait()

# 更新波形图
def update_plot():
    ax.clear()
    ax.plot(attention_values, color='blue')
    ax.set_ylim([0, 100])
    ax.set_title("实时注意力波形图")
    ax.set_xlabel("时间")
    ax.set_ylabel("注意力水平")
    canvas.draw()

# 启动监测线程
def start_monitoring():
    global stop_monitoring
    stop_monitoring = False
    threading.Thread(target=monitor_and_plot).start()

# 停止监测
def stop_monitoring_action():
    global stop_monitoring
    stop_monitoring = True

# 设置低注意力警告阈值
attention_threshold = 30  # 低于此值触发警告

# 创建主窗口
root = tk.Tk()
root.title("EEG 注意力监测系统")

# 显示注意力值的标签
attention_var = tk.StringVar()
attention_label = tk.Label(root, textvariable=attention_var, font=("Helvetica", 16))
attention_label.pack(pady=20)

# 创建波形图
fig, ax = plt.subplots(figsize=(6, 4))
attention_values = []
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# 开始按钮
start_button = tk.Button(root, text="开始监测", command=start_monitoring, font=("Helvetica", 14))
start_button.pack(pady=10)

# 停止按钮
stop_button = tk.Button(root, text="停止监测", command=stop_monitoring_action, font=("Helvetica", 14))
stop_button.pack(pady=10)

# 初始化停止监测标志
stop_monitoring = False

# 启动 GUI
root.mainloop()
