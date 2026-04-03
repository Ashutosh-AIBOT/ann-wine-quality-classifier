from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from path_utils import DATA_PROCESSED, MODELS


class WineANN(nn.Module):
    def __init__(self, input_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

_cache = {}


def load_model_and_scaler() -> tuple[WineANN | None, any | None]:
    if _cache:
        return _cache.get("model"), _cache.get("scaler")

    try:
        scaler_path = DATA_PROCESSED / "scaler.pkl"
        if not scaler_path.exists():
            print(f"❌ Error: {scaler_path} missing.")
            return None, None
            
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        model_path = MODELS / "model.pkl"
        if not model_path.exists():
            print(f"❌ Error: {model_path} missing.")
            return None, None

        model = WineANN(input_dim=11)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        _cache["model"] = model
        _cache["scaler"] = scaler
        return model, scaler
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None, None


def predict_class(features_dict: dict):
    missing = [k for k in FEATURES if k not in features_dict]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    model, scaler = load_model_and_scaler()
    frame = pd.DataFrame([{k: features_dict[k] for k in FEATURES}])
    x = scaler.transform(frame)
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1).numpy().reshape(-1)
    pred = int(np.argmax(probs))
    return pred, probs.tolist()


def get_training_history() -> dict:
    path = DATA_PROCESSED / "training_history.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_classification_report() -> dict:
    path = MODELS / "results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data
