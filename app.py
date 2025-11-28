HEAD
import logging

=======
import os
import joblib
import pandas as pd
import streamlit as st

HEAD
from src.config import DATA_PATH, MODEL_PATH
from src.train import train_model

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data + model loading
# -------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame | None:
    try:
        df = pd.read_csv(DATA_PATH)
        unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)
        return df
    except FileNotFoundError:
        logger.warning("Data file not found at %s", DATA_PATH)
        return None

=======
MODEL_PATH = os.path.join("models", "house_price_model.pkl")
DATA_PATH = "Mumbai1.csv"


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        return df
    return None
>>>>>>> 4a14772 (Local changes before rebase)


@st.cache_resource
def load_model():
HEAD
    """
    Load trained model.

    obj = joblib.load(MODEL_PATH)
    pipeline = obj["pipeline"]
    numeric_features = obj["numeric_features"]
    categorical_features = obj["categorical_features"]
    return pipeline, numeric_features, categorical_features

4a14772 (Local changes before rebase)

    If the serialized model is missing or incompatible in the cloud
    environment, retrain and then reload.
    """
    logger.info("Loading model from %s", MODEL_PATH)
    try:
        obj = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.warning("Model load failed (%s). Retraining...", e)
        obj = train_model()

    pipeline = obj["pipeline"]
    numeric_features = obj["numeric_features"]
    categorical_features = obj["categorical_features"]
    return pipeline, numeric_features, categorical_features


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Mumbai Real Estate Price Prediction", layout="wide")

    st.title("Mumbai Real Estate Price Prediction")
    st.write(
        "Estimate property prices in Mumbai using a machine learning model "
        "trained on historical transaction data."
    )

    df = load_data()
    model, numeric_features, categorical_features = load_model()

HEAD
    if df is not None and "Location" in df.columns:
        locations = sorted(df["Location"].dropna().unique().tolist())
    else:
        locations = ["Mumbai"]

    st.header("Property Details")

    if df is not None:
        locations = sorted(df["Location"].dropna().unique().tolist())
    else:
        locations = []

    st.subheader("Property details")
4a14772 (Local changes before rebase)

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq ft)", min_value=100.0, max_value=10000.0, value=800.0, step=50.0
        )
        bedrooms = st.number_input(
            "No. of Bedrooms", min_value=1, max_value=10, value=2, step=1
        )
HEAD
        location = st.selectbox("Location", options=locations)

        location = st.selectbox(
            "Location",
            options=locations if locations else ["Kharghar"],
        )
4a14772 (Local changes before rebase)

    with col2:
        new_resale = st.selectbox("New or Resale", ["Resale", "New"])
        gym = st.checkbox("Gymnasium")
        lift = st.checkbox("Lift Available", value=True)
        parking = st.checkbox("Car Parking", value=True)
        maint = st.checkbox("Maintenance Staff", value=True)
        sec24 = st.checkbox("24x7 Security", value=True)
        play = st.checkbox("Children's Play Area")
        club = st.checkbox("Clubhouse")
        intercom = st.checkbox("Intercom")
        garden = st.checkbox("Landscaped Gardens")
        indoor = st.checkbox("Indoor Games")
        gas = st.checkbox("Gas Connection")
        track = st.checkbox("Jogging Track")
        pool = st.checkbox("Swimming Pool")

    row = {
        "Area": area,
        "No. of Bedrooms": bedrooms,
        "New/Resale": 1 if new_resale == "New" else 0,
        "Gymnasium": int(gym),
        "Lift Available": int(lift),
        "Car Parking": int(parking),
        "Maintenance Staff": int(maint),
        "24x7 Security": int(sec24),
        "Children's Play Area": int(play),
        "Clubhouse": int(club),
        "Intercom": int(intercom),
        "Landscaped Gardens": int(garden),
        "Indoor Games": int(indoor),
        "Gas Connection": int(gas),
        "Jogging Track": int(track),
        "Swimming Pool": int(pool),
        "Location": location,
    }

    feature_cols = numeric_features + categorical_features
    X_input = pd.DataFrame([row])[feature_cols]

    if st.button("Predict Price"):
        price = model.predict(X_input)[0]
        st.subheader(f"Estimated Price: ₹{price:,.0f}")

        if df is not None and "Price" in df.columns:
            avg_price = df["Price"].mean()
            st.caption(f"Average price in dataset: ₹{avg_price:,.0f}")
HEAD

        st.caption("Note: This tool is for educational use only, not financial advice.")

4a14772 (Local changes before rebase)



if __name__ == "__main__":
    main()

