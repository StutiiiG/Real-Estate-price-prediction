# 🏙️ Mumbai Real Estate Price Predictor 

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-brightgreen?logo=streamlit)](https://realestatepricepredictormumbai.streamlit.app)

An interactive machine learning web application that predicts housing prices across Mumbai based on property characteristics such as area, locality, amenities, and property condition.

## 🚀 Live Demo
https://realestatepricepredictormumbai.streamlit.app

---

## 📸 App Previews

| Home UI | Price Output |
|--------|--------------|
| ![UI Screenshot 1](./assets/UI_home.png) | ![UI Screenshot 2](./assets/UI_prediction.png) |

---

## 🎯 Key Features

| Feature | Description |
|--------|-------------|
| Automated ML Pipeline | Preprocessing, feature encoding, and model training |
| Live Deployment | Real-time inference via Streamlit UI |
| Location Intelligence | Captures value differences across Mumbai neighborhoods |
| Configurable Property Features | Bedrooms, area, amenities, property type |
| Cloud-Safe Execution | Retrains in cloud to ensure compatibility |

---

## 🧠 Model Workflow

User Input → Feature Engineering → RandomForestRegressor → Price Prediction → UI Rendering

The serialized trained pipeline is version-controlled for reproducibility and efficient loading.

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|------------|
| Machine Learning | Python, Scikit-Learn, Pandas |
| Web UI | Streamlit |
| Model Persistence | Joblib |
| Deployment | Streamlit Cloud |
| Version Control | Git + GitHub |

---




