import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")

st.title("已训练模型 EEG 多模型预测对比平台")

channels = ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta', 'low_gamma', 'mid_gamma']

#  文件上传和预览
uploaded_file = st.file_uploader("上传 CSV 文件", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("数据预览")
    st.dataframe(df.head())

    seq_len = 10  # 时间步保持和训练一致
    test_split = 0.2

    # 选择通道
    selected_channel = st.selectbox("选择要对比的波段", channels)
    channel_idx = channels.index(selected_channel)

    # 选择模型
    selected_models = st.multiselect("选择模型", ["ESN", "LSTM", "Transformer"], default=["ESN", "LSTM", "Transformer"])

    if st.button("开始预测"):
        with st.spinner("预测中..."):

            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_csv_path = tmp_file.name

            # 数据预处理
            df = pd.read_csv(tmp_csv_path)
            df = df.drop(columns=['timestamp'], errors='ignore')
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df)

            # 构建序列
            X, y = [], []
            for i in range(len(scaled) - seq_len):
                X.append(scaled[i:i + seq_len])
                y.append(scaled[i + seq_len])
            X = np.array(X)
            y = np.array(y)

            # 拆分测试集
            split_idx = int(len(X) * (1 - test_split))
            X_test = X[split_idx:]
            y_test = y[split_idx:]

            results = {}

            # ESN预测
            if "ESN" in selected_models:
                reservoir = joblib.load("saved_models/esn_reservoir.pkl")
                readout = joblib.load("saved_models/esn_readout.pkl")
                X_esn = X_test.reshape(X_test.shape[0], -1) # X_esn 被展平为二维数组。
                y_pred_esn = readout.run(reservoir.run(X_esn))
                results['ESN'] = y_pred_esn

            # LSTM预测
            if "LSTM" in selected_models:
                lstm_model = load_model("saved_models/lstm_model.h5")
                y_pred_lstm = lstm_model.predict(X_test)
                results['LSTM'] = y_pred_lstm

            # Transformer预测
            if "Transformer" in selected_models:
                trans_model = load_model("saved_models/transformer_model.h5")
                y_pred_trans = trans_model.predict(X_test)
                results['Transformer'] = y_pred_trans

        st.success("预测完成!")

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_test[:, channel_idx], label='True', linewidth=2)
        for model_name, y_pred in results.items():
            ax.plot(y_pred[:, channel_idx], label=model_name, linestyle='--')
        ax.set_title(f"{selected_channel} 波段预测对比")
        ax.legend()
        st.pyplot(fig)

        # 显示误差指标
        st.subheader("MSE / MAE 指标")

        metrics = {
            "模型": [],
            "MSE": [],
            "MAE": []
        }

        for model_name, y_pred in results.items():
            mse = mean_squared_error(y_test[:, channel_idx], y_pred[:, channel_idx])
            mae = mean_absolute_error(y_test[:, channel_idx], y_pred[:, channel_idx])
            metrics["模型"].append(model_name)
            metrics["MSE"].append(mse)
            metrics["MAE"].append(mae)

        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df)

        # 下载预测结果
        st.subheader("下载预测结果")
        for model_name, y_pred in results.items():
            pred_df = pd.DataFrame(y_pred, columns=channels)
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(f"下载 {model_name} 预测结果", csv, file_name=f"{model_name}_prediction.csv", mime='text/csv')
