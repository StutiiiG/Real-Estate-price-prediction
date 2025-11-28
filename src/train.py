import logging
from typing import Dict, List

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_DIR, MODEL_PATH
from .data_utils import load_raw_data, split_features_target, train_test_split

logger = logging.getLogger(__name__)


def build_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    """
    Build a preprocessing + model pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model() -> Dict:
    """
    Train the model pipeline on the Mumbai housing dataset,
    evaluate, and persist the trained pipeline + metadata.

    Returns:
        dict with keys: "pipeline", "numeric_features", "categorical_features"
    """
    logger.info("Starting model training pipeline")

    df = load_raw_data()
    train_df, test_df = train_test_split(df)

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    # Infer numeric vs categorical features from dtypes
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        c for c in X_train.columns if c not in numeric_features
    ]

    logger.info("Numeric features: %s", numeric_features)
    logger.info("Categorical features: %s", categorical_features)

    pipeline = build_pipeline(numeric_features, categorical_features)

    logger.info("Fitting model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info("MAE: %.2f", mae)
    logger.info("R^2: %.3f", r2)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    obj = {
        "pipeline": pipeline,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": {"mae": mae, "r2": r2},
    }
    joblib.dump(obj, MODEL_PATH)
    logger.info("Saved trained model to %s", MODEL_PATH)

    return obj


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    train_model()
