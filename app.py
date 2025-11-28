import os
import joblib
import pandas as pd
import streamlit as st
from model_training import train_model

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

@st.cache_resource
def load_model():
    """
    Load the trained model. If the pickle is missing or incompatible
    (e.g., on Streamlit Cloud), retrain the model from Mumbai1.csv.
    """
    # 1) If file doesn't exist at all, train first
    if not os.path.exists(MODEL_PATH):
        train_model()

    # 2) Try to load existing pickle
    try:
        obj = joblib.load(MODEL_PATH)
        pipeline = obj["pipeline"]
        numeric_features = obj["numeric_features"]
        categorical_features = obj["categorical_features"]
        return pipeline, numeric_features, categorical_features
    except Exception:
        # 3) If load fails (corrupt / incompatible), retrain then load again
        train_model()
        obj = joblib.load(MODEL_PATH)
        pipeline = obj["pipeline"]
        numeric_features = obj["numeric_features"]
        categorical_features = obj["categorical_features"]
        return pipeline, numeric_features, categorical_features


def main():
    st.title("Mumbai Real Estate Price Prediction")
    st.write(
        "Enter property details below to get an estimated price based on a machine "
        "learning model trained on Mumbai housing data."
    )

    df = load_data()
    model, numeric_features, categorical_features = load_model()

    if df is not None:
        locations = sorted(df["Location"].dropna().unique().tolist())
    else:
        locations = []

    st.subheader("Property details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq ft)", min_value=100.0, max_value=10000.0, value=800.0, step=50.0
        )
        bedrooms = st.number_input(
            "No. of Bedrooms", min_value=1, max_value=10, value=2, step=1
        )
        location = st.selectbox(
            "Location",
            options=locations if locations else ["Kharghar"],
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

        st.caption("Note: This is an educational demo, not financial advice.")


if __name__ == "__main__":
    main()
