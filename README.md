# 🏙️ Mumbai Real Estate Price Prediction

[![Streamlit App](https://img.shields.io/badge/🚀_Live%20Demo-Streamlit-red?logo=streamlit)](https://realestatepricepredictormumbai.streamlit.app)

A machine learning web app to estimate **Mumbai property prices** based on area, locality, amenities, and housing features. Predictions are made using a trained regression model powered by **Scikit-Learn**, wrapped into an interactive app using **Streamlit**, and deployed on **Streamlit Cloud**.

🔗 **Live App:** https://realestatepricepredictormumbai.streamlit.app  
📂 **GitHub Repo:** https://github.com/StutiiiG/Real-Estate-price-prediction

---

## 📸 App Preview

> *(Note: Add a screenshot to the project at `notebooks/app_preview.png` for this image to show)*

<p align="center">
  <img src="https://raw.githubusercontent.com/StutiiiG/Real-Estate-price-prediction/main/notebooks/app_preview.png" width="80%">
</p>

---
## ✨ Key Features

✔ Predicts house prices instantly  
✔ Location-based estimates for multiple Mumbai regions  
✔ Specify amenities like:
- Gymnasium  
- Car Parking  
- 24×7 Security  
- Lift  
✔ Includes property type (New vs Resale)  
✔ User-friendly interface — mobile & desktop responsive

---
## 🛠 Tech Stack

| Layer | Tools |
|------|------|
| Frontend UI | Streamlit |
| Machine Learning | Scikit-Learn, Pandas, NumPy |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 📂 Project Structure

```text
.
├── data/
│   └── Mumbai1.csv               # raw dataset
├── models/
│   └── house_price_model.pkl     # trained regression pipeline
├── notebooks/
│   └── real_estate_model.ipynb   # exploratory analysis & experimentation
├── app.py                        # Streamlit app
├── model_training.py             # training script
├── requirements.txt
└── README.md

## ▶️ How to Run Locally

```bash
git clone https://github.com/StutiiiG/Real-Estate-price-prediction.git
cd Real-Estate-price-prediction
pip install -r requirements.txt
streamlit run app.py
