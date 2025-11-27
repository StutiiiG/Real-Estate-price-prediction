import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = "Mumbai1.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)

    # === TODO: UPDATE THESE TO MATCH YOUR CSV COLUMN NAMES ===
    # Example only – REPLACE with your actual names from df.columns
    feature_cols = ["area", "bedrooms", "bathrooms"]  # <-- change
    target_col = "price"                               # <-- change
    # ========================================================

    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def train_model():
    X, y, feature_cols = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"R² : {r2:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": feature_cols}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
