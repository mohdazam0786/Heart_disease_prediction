# ❤️ Heart Disease Prediction App

A machine learning project that predicts the likelihood of heart disease based on patient health data. This project includes **EDA, preprocessing, model comparison, and deployment using Streamlit**.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. This project aims to:

- Analyze heart disease data
- Build multiple ML models
- Compare their performance
- Deploy the best model using a user-friendly web app

---

## 📊 Dataset

- File: `heart.csv`
- Target Variable: `HeartDisease`  
  - `0` → No Disease  
  - `1` → Heart Disease  

### Features include:
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- ECG Results  
- Max Heart Rate  
- Exercise Angina  
- Oldpeak  
- ST Slope  

---

## 🔍 Exploratory Data Analysis (EDA)

Performed:
- Data shape, info, and summary statistics
- Missing values and duplicates check
- Distribution plots (Age, BP, Cholesterol, MaxHR)
- Count plots for categorical variables
- Boxplot & violin plot for relationships
- Correlation heatmap

---

## 🧹 Data Preprocessing

- Replaced `0` values in:
  - Cholesterol → mean (excluding zeros)
  - RestingBP → mean (excluding zeros)
- One-hot encoding using `pd.get_dummies()`
- Converted all features to numeric
- Feature scaling using `StandardScaler`

---

## 🤖 Models Used

The following models were trained and evaluated:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Decision Tree  
- Support Vector Machine (SVM)  

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy Score  
- F1 Score  

Example results:

| Model | Accuracy | F1 Score |
|------|--------|----------|
| Logistic Regression | ~0.87 | ~0.88 |
| KNN | ~0.88 | ~0.89 |
| SVM | ~0.86 | ~0.88 |

👉 **KNN performed best and was selected for deployment**

---

## 💾 Saved Files

- `KNN_heart.pkl` → Trained model  
- `scaler.pkl` → StandardScaler  
- `columns.pkl` → Feature columns after encoding  

---

## 🌐 Streamlit App

A simple web interface where users can:

- Input health parameters  
- Get instant prediction  
- View risk level (High / Low)  

---
