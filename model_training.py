import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = "Mumbai1.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)

    # drop useless index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    target_col = "Price"

    # feature columns based on your CSV
    numeric_features = [
        "Area",
        "No. of Bedrooms",
        "New/Resale",
        "Gymnasium",
        "Lift Available",
        "Car Parking",
        "Maintenance Staff",
        "24x7 Security",
        "Children's Play Area",
        "Clubhouse",
        "Intercom",
        "Landscaped Gardens",
        "Indoor Games",
        "Gas Connection",
        "Jogging Track",
        "Swimming Pool",
    ]

    categorical_features = ["Location"]

    X = df[numeric_features + categorical_features]
    y = df[target_col]

    return X, y, numeric_features, categorical_features


def train_model():
    X, y, numeric_features, categorical_features = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",  # numeric features go through as-is
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:,.2f}")
    print(f"R^2: {r2:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
