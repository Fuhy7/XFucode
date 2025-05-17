# app.py

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

from model_loader import load_all_models
from old_ui.predictor import run_prediction
from data_utils import load_eeg_csv

import traceback
import dash_html_components as html

app = dash.Dash(__name__)
app.title = "EEG 多模型预测对比仪表板"

channels = ['delta', 'theta', 'low_alpha', 'high_alpha',
            'low_beta', 'high_beta', 'low_gamma', 'mid_gamma']

# 页面布局
app.layout = html.Div([
    html.H2("🧠 EEG 模型预测对比仪表板 v2"),

    html.Label("选择波段通道:"),
    dcc.Dropdown(
        id='channel-dropdown',
        options=[{"label": ch, "value": ch} for ch in channels],
        value="mid_gamma",
        style={"width": "200px"}
    ),

    html.Button("执行预测", id="predict-button", n_clicks=0),
    html.Br(), html.Br(),

    dcc.Graph(id='prediction-graph'),

    html.Div(id='status')
])

def safe_callback(callback_func):
    def wrapped_callback(*args, **kwargs):
        try:
            return callback_func(*args, **kwargs)
        except Exception as e:
            print("🚨 Dash 回调发生异常:")
            traceback.print_exc()

            # 返回空图表和前端错误提示
            empty_fig = {
                "data": [],
                "layout": {
                    "title": "⚠️ 发生错误",
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False}
                }
            }

            error_display = html.Div([
                html.P("❌ 错误信息：", style={"color": "red", "fontWeight": "bold"}),
                html.Pre(str(e), style={"color": "red", "whiteSpace": "pre-wrap"})
            ])

            return empty_fig, error_display
    return wrapped_callback


@app.callback(
    Output("prediction-graph", "figure"),
    Output("status", "children"),
    Input("predict-button", "n_clicks"),
    State("channel-dropdown", "value")
)
@safe_callback
def update_prediction(n_clicks, selected_channel):
    if n_clicks == 0:
        return go.Figure(), ""

    df = load_eeg_csv("../eeg_brainwaves_1hour1.csv")
    models = load_all_models()
    channel_idx = df.columns.get_loc(selected_channel)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df[selected_channel], name="真实值"))

    error_rows = []

    for model_name, model_obj in models.items():
        # y_true, y_pred, _, _ = run_prediction(
        #     model_obj['model'], df, model_obj['type'], selected_channel
        # )
        y_true, y_pred, mse, mae = run_prediction(
            model_obj['model'],
            df,
            model_obj['type'],
            selected_channel,
            model_name
        )

        print(f"{model_name} → y_pred.shape = {y_pred.shape}")
        print(f"{model_name} → y_pred[:5] = {y_pred[:5]}")

        # ✅ 多通道提取对应通道列
        fig.add_trace(go.Scatter(
            y=y_pred,  # 已在 predictor.py 中选好了通道
            name=f"{model_name}预测"
        ))

        # ✅ 误差计算
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        # mse = mean_squared_error(y_true[:, channel_idx], y_pred[:, channel_idx])
        # mae = mean_absolute_error(y_true[:, channel_idx], y_pred[:, channel_idx])
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        error_rows.append(html.Tr([
            html.Td(model_name), html.Td(f"{mse:.2f}"), html.Td(f"{mae:.2f}")
        ]))

    fig.update_layout(title=f"{selected_channel.upper()} - 三模型预测对比", height=500)

    table = html.Div([
        html.H4("📊 模型误差对比表"),
        html.Table([
            html.Tr([html.Th("模型"), html.Th("MSE"), html.Th("MAE")]),
            *error_rows
        ], style={"width": "50%", "border": "1px solid #ccc", "textAlign": "center"})
    ])

    return fig, table



if __name__ == '__main__':
    app.run_server(debug=True)
