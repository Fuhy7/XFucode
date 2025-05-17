import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64
import io
import tensorflow as tf
import joblib
from scipy.fftpack import fft
from tensorflow.keras.initializers import OrthogonalInitializer  # 导入初始化器

# 自定义 InputLayer 类，忽略 batch_shape 参数
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs.pop('batch_shape')
        super().__init__(**kwargs)

# 初始化 Dash 应用
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 加载模型并编译
transformer_model = tf.keras.models.load_model("model/eeg_attention_transformer_model.h5")
transformer_model.compile(optimizer='adam', loss='mse')

# 加载 LSTM 模型时使用自定义对象
custom_objects = {
    'InputLayer': CustomInputLayer,
    'OrthogonalInitializer': OrthogonalInitializer  # 添加到自定义对象中
}
lstm_model = tf.keras.models.load_model("model/lstm_eeg_model.h5", custom_objects=custom_objects)
lstm_model.compile(optimizer='adam', loss='mse')

esn_model = joblib.load("model/esn_eeg_model.pkl")

# 生成模拟 EEG 数据（每个输入为 10 个时间步）
time_series = np.linspace(0, 10, 1000)
eeg_signal = np.sin(2 * np.pi * 10 * time_series) + np.random.normal(0, 0.5, len(time_series))

# 获取 LSTM/SimpleRNN 模型的输入形状
feature_dim = 6  # 确保 LSTM 和 SimpleRNN 需要的输入维度一致


# 计算 FFT 频域特征，并融合时间域特征
def extract_features(data):
    fft_values = np.abs(fft(data, n=10))[:3]  # 取前三个 FFT 频段
    fft_values = fft_values / np.max(fft_values)  # 归一化

    time_features = np.array([
        np.mean(data), np.std(data), np.min(data)  # 只选 3 个时间特征
    ])

    features = np.concatenate([fft_values, time_features])  # 组合时间域+频域特征
    assert features.shape[0] == 6, "特征维度错误，应为 6 维"
    return features


# 预测模型函数
def predict_eeg(data, model_type):
    if len(data) < 10:
        return data  # 数据不足，无法预测

    X_input = np.array([extract_features(data[i:i + 10]) for i in range(len(data) - 10)])
    X_input = X_input.reshape(-1, 10, feature_dim)  # 确保形状为 (batch_size, 10, 6)

    print("FFT 特征样本: ", X_input[0])  # 调试查看输入特征

    if model_type == "Transformer Model":
        predictions = transformer_model.predict(X_input)
    elif model_type == "LSTM Model":
        predictions = lstm_model.predict(X_input)
    elif model_type == "ESN Model":
        X_input = X_input.reshape(X_input.shape[0], -1)
        predictions = esn_model.predict(X_input)
    else:
        return data

    predictions = predictions.flatten()
    predictions = np.concatenate([data[:10], predictions])
    return predictions[:len(data)]  # 确保返回数据长度匹配


# Dash 页面布局
app.layout = dbc.Container([
    html.H1("EEG 信号时序预测", className="text-center mt-4"),

    dcc.Upload(
        id='upload-data',
        children=html.Button("上传 EEG 数据（CSV）", className="btn btn-primary"),
        multiple=False
    ),

    dbc.Row([
        dbc.Col(html.Label("选择预测模型：")),
        dbc.Col(dcc.Dropdown(
            id='model-selector',
            options=[
                {'label': 'Transformer Model', 'value': 'Transformer Model'},
                {'label': 'LSTM Model', 'value': 'LSTM Model'},
                {'label': 'ESN Model', 'value': 'ESN Model'}
            ],
            value='Transformer Model'
        ))
    ], className="mt-3"),

    dbc.Row([
        dbc.Col(html.Label("选择回放时间段：")),
        dbc.Col(dcc.RangeSlider(
            id='time-slider',
            min=0,
            max=len(time_series),
            step=10,
            value=[0, 200],
            marks={i: str(i) for i in range(0, len(time_series), 200)}
        ))
    ], className="mt-3"),

    dcc.Graph(id='eeg-graph'),
    html.H4("预测结果", className="mt-4"),
    dcc.Graph(id='prediction-graph'),
])


@app.callback(
    [Output('eeg-graph', 'figure'),
     Output('prediction-graph', 'figure')],
    [Input('time-slider', 'value'),
     Input('model-selector', 'value')],
    [State('upload-data', 'contents')]
)
def update_graphs(time_range, model_type, uploaded_file):
    global eeg_signal

    if uploaded_file:
        content_type, content_string = uploaded_file.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        eeg_signal = df.iloc[:, 1].values  # 假设 EEG 数据在 CSV 文件的第二列

    start, end = time_range
    x_data = time_series[start:end]
    y_data = eeg_signal[start:end]

    if len(y_data) < 10:
        prediction_data = y_data  # 数据过短，不做预测
    else:
        prediction_data = predict_eeg(y_data, model_type)

    if len(prediction_data) != len(y_data):
        prediction_data = np.pad(prediction_data, (0, len(y_data) - len(prediction_data)), mode='edge')

    eeg_fig = go.Figure()
    eeg_fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='EEG 信号'))
    eeg_fig.update_layout(title="EEG 时序数据", xaxis_title="时间", yaxis_title="信号强度")

    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='原始信号'))
    pred_fig.add_trace(go.Scatter(x=x_data, y=prediction_data, mode='lines', name='预测信号'))
    pred_fig.update_layout(title="预测结果", xaxis_title="时间", yaxis_title="信号强度")

    return eeg_fig, pred_fig


if __name__ == '__main__':
    app.run_server(debug=True)