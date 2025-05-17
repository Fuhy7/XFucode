# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import MinMaxScaler
# from data_utils import create_sequences
#
# def run_prediction(model, df, model_type, target_col, seq_len=10):
#     data = df.values
#     target_idx = df.columns.get_loc(target_col)
#
#     # 多通道输入和输出
#     X, y_all = create_sequences(data, seq_len=seq_len)  # y_all: [N, 8]
#
#     if model_type == "lstm_or_transformer":
#         y_pred_all = model.predict(X)  # shape: [N, 8]
#     elif model_type == "esn":
#         reservoir, readout = model
#         X_flat = X.reshape(X.shape[0], -1)
#         y_pred_all = readout.run(reservoir.run(X_flat))
#
#     # 防御性 reshape（避免 1D）
#     if y_pred_all.ndim == 1:
#         y_pred_all = y_pred_all.reshape(-1, 1)
#     if y_all.ndim == 1:
#         y_all = y_all.reshape(-1, 1)
#
#     min_len = min(len(y_all), len(y_pred_all))
#     y_true = y_all[:min_len, target_idx]
#     y_pred = y_pred_all[:min_len, target_idx]
#
#     # 反归一化（只对目标通道）
#     scaler = MinMaxScaler()
#     scaler.fit(df.values)
#     min_val = scaler.data_min_[target_idx]
#     max_val = scaler.data_max_[target_idx]
#     unnormalize = lambda x: x * (max_val - min_val) + min_val
#
#     y_true = unnormalize(y_true)
#     y_pred = unnormalize(y_pred)
#
#     # 误差
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#
#     return y_true, y_pred, mse, mae

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from data_utils import create_sequences

def run_prediction(model, df, model_type, target_col, model_name, seq_len=10):
    data = df.values
    target_idx = df.columns.get_loc(target_col)

    X, y_all = create_sequences(data, seq_len=seq_len)

    # 训练时统一 fit scaler
    scaler = MinMaxScaler()
    scaler.fit(df.values)

    min_val = scaler.data_min_[target_idx]
    max_val = scaler.data_max_[target_idx]
    unnormalize = lambda x: x * (max_val - min_val) + min_val

    # 模型预测
    if model_type == "lstm_or_transformer":
        y_pred_all = model.predict(X)

        # Transformer 反归一化修复
        if "transformer" in model_name.lower():
            if y_pred_all.ndim == 1:
                y_pred_all = y_pred_all.reshape(-1, 1)  # 确保二维
            y_pred_all = scaler.inverse_transform(y_pred_all)

    elif model_type == "esn":
        reservoir, readout = model
        X_flat = X.reshape(X.shape[0], -1)
        y_pred_all = readout.run(reservoir.run(X_flat))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if y_pred_all.ndim == 1:
        y_pred_all = y_pred_all.reshape(-1, 1)
    if y_all.ndim == 1:
        y_all = y_all.reshape(-1, 1)

    min_len = min(len(y_all), len(y_pred_all))
    y_true = y_all[:min_len, target_idx]
    y_pred = y_pred_all[:min_len, target_idx]

    # 反归一化
    if "transformer" in model_name.lower():
        # transformer 已经 inverse_transform 过，无需再次 unnormalize
        pass
    else:
        # LSTM 和 ESN 使用 lambda 反归一化
        y_true = unnormalize(y_true)
        y_pred = unnormalize(y_pred)

    print(f"{model_name} → y_pred[:5] = {y_pred[:5]}")

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return y_true, y_pred, mse, mae

