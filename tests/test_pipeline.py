import pandas as pd

from src.config import MODEL_PATH
from src.train import train_model
import joblib


def test_model_trains_and_saves(tmp_path, monkeypatch):
    """
    Smoke test: model can be trained and serialized without error.
    """
    # Override MODEL_PATH to avoid overwriting real model
    from src import config as cfg

    test_model_path = tmp_path / "test_model.pkl"
    monkeypatch.setattr(cfg, "MODEL_PATH", test_model_path)

    obj = train_model()
    assert "pipeline" in obj
    assert test_model_path.exists()


def test_model_prediction_shape():
    """
    Smoke test: loaded model can produce a single numeric prediction.
    """
    obj = None
    if MODEL_PATH.exists():
        obj = joblib.load(MODEL_PATH)
    else:
        obj = train_model()

    pipeline = obj["pipeline"]

    # Minimal fake row with required columns
    sample = pd.DataFrame(
        [
            {
                "Area": 800,
                "No. of Bedrooms": 2,
                "New/Resale": 1,
                "Gymnasium": 1,
                "Lift Available": 1,
                "Car Parking": 1,
                "Maintenance Staff": 1,
                "24x7 Security": 1,
                "Children's Play Area": 0,
                "Clubhouse": 0,
                "Intercom": 0,
                "Landscaped Gardens": 0,
                "Indoor Games": 0,
                "Gas Connection": 0,
                "Jogging Track": 0,
                "Swimming Pool": 0,
                "Location": "Mumbai",
            }
        ]
    )

    y_pred = pipeline.predict(sample)
    assert y_pred.shape == (1,)
    assert float(y_pred[0]) > 0
