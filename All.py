# model_compare.py

import pandas as pd
from LSTM_run_optimized import main as run_lstm
from ESN_run_optimized import main as run_esn
from Transformer_run_optimized import main as run_transformer


# 将每个模型主函数改为返回误差值（或手动加上）
# 返回格式： mse, mae


def run_all_models():
    results = []

    # === Run LSTM ===
    print("Running LSTM...")
    mse, mae = run_lstm(return_metrics=True)  # 你需要把 run_lstm 的 main 函数支持返回这两个指标
    results.append({"Model": "LSTM", "MSE": mse, "MAE": mae})

    # === Run ESN ===
    print(" Running ESN...")
    mse, mae = run_esn(return_metrics=True)
    results.append({"Model": "ESN", "MSE": mse, "MAE": mae})

    # === Run Transformer ===
    print("🔗 Running Transformer...")
    mse, mae = run_transformer(return_metrics=True)
    results.append({"Model": "Transformer", "MSE": mse, "MAE": mae})

    # === Save results ===
    df = pd.DataFrame(results)
    print("\n Final Results:")
    print(df)

    df.to_csv("model_compare_results.csv", index=False)
    print("\nSaved to model_compare_results.csv")


if __name__ == "__main__":
    run_all_models()
