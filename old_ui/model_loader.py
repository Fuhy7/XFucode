from tensorflow.keras.models import load_model
import joblib

def load_all_models():
    return {
        "LSTM": {
            "model": load_model("saved_models/lstm_model.h5"),
            "type": "lstm_or_transformer"
        },
        "Transformer": {
            "model": load_model("saved_models/transformer_model.h5"),
            "type": "lstm_or_transformer"
        },
        "ESN": {
            "model": (
                joblib.load("../saved_models/esn_reservoir.pkl"),
                joblib.load("../saved_models/esn_readout.pkl")
            ),
            "type": "esn"
        }
    }
