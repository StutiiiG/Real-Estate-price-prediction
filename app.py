import logging
import joblib
import pandas as pd
import streamlit as st

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
# Data loading
# -------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Load the Mumbai housing dataset from DATA_PATH."""
    try:
        df = pd.read_csv(DATA_PATH)

        # Drop unnamed index columns if present
        unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)

        return df
    except FileNotFoundError:
        logger.warning("Data file not found at %s", DATA_PATH)
        return None
    except Exception as e:
        logger.exception("Unexpected error while loading data: %s", e)
        return None


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Load the trained model pipeline.

    If the serialized model is missing or incompatible in the cloud
    environment, retrain and then reload.
    """
    logger.info("Loading model from %s", MODEL_PATH)
    try:
        obj = joblib.load(MODEL_PATH)
    except Exception as e:
        logger.warning("Model load failed (%s). Retraining model...", e)
        obj = train_model()

    pipeline = obj["pipeline"]
    numeric_features = obj["numeric_features"]
    categorical_features = obj["categorical_features"]
    return pipeline, numeric_features, categorical_features


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Mumbai Real Estate Price Prediction",
        layout="wide",
    )

    st.title("Mumbai Real Estate Price Prediction")
    st.write(
        "Enter property details below to get an estimated price for a property in Mumbia, India"
    )

    df = load_data()
    model, numeric_features, categorical_features = load_model()

    # ----- Location options -----
    if df is not None and "Location" in df.columns:
        locations = sorted(df["Location"].dropna().unique().tolist())
    else:
        locations = ["Mumbai"]

    # ----- Input form -----
    st.header("Property details")
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq ft)",
            min_value=100.0,
            max_value=10000.0,
            value=800.0,
            step=50.0,
        )
        bedrooms = st.number_input(
            "No. of Bedrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
        )
        location = st.selectbox(
            "Location",
            options=locations,
        )

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

    # Build single-row input for the model
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

    st.caption("Note: This tool is for educational use only, not financial advice.")


if __name__ == "__main__":
    main()


