# 📈 US Stocks Financial Indicator Analysis (2014–2018)

This project involves a comprehensive financial analysis and machine learning modeling on a dataset of US stocks from 2014 to 2018. It includes feature engineering, classification, regression, data leakage handling, and evaluation of important features influencing stock price movements.

---

## 📂 Dataset

- **Source**: [Kaggle - 200 Financial Indicators of US Stocks 2014–2018](https://www.kaggle.com/datasets)
- **Records**: 22,077 rows
- **Features**: 228 financial indicators per stock per year
- **Target Variables**:
  - `Class`: Binary label indicating stock performance (classification)
  - `2019 PRICE VAR [%]`: Price variation used for regression & future forecasting

---

## 🧹 Data Cleaning & Preparation

- Removed duplicates
- Handled missing values using `SimpleImputer` with `median` or `mean` strategy
- Dropped high-leakage columns like `Class`, `PRICE VAR [%]` for fair training
- Categorical features (like sector) were removed before numeric imputation

---

## 🔍 Exploratory Data Analysis (EDA)

- Distribution of stock performance across years
- Sector-wise analysis of key financial metrics
- Correlation heatmaps to detect multicollinearity
- Feature importance through model explainability

---

## 🤖 Machine Learning Models

### 1. **Classification (Binary: Good/Bad Stock)**
- **Logistic Regression**
- **Random Forest Classifier**
- **Handling Data Leakage** by removing future price-based features

📊 Sample Evaluation:

```
✅ Accuracy: 0.998
Confusion Matrix:
[[1968    1]
 [   7 2440]]
F1-score: ~1.00
```

Top 10 Predictive Features:
- `Class`, `PRICE VAR [%]`, `EPS`, `Revenue`, `Payout Ratio`, etc.

---

### 2. **Regression (Predicting 2019 Price Variance)**
- **Random Forest Regressor**
- Trained only on 2018 data to avoid forward-looking bias

📉 Mean Squared Error (MSE): ~20,000  
📈 R² Score: ~-0.004 (due to outliers and noisy financial forecasting)

Top 10 Features:
- `Gross Profit`, `Cash Flow Ratios`, `Earnings Yield`, `R&D Growth`

---

### 3. **Converted Regression into Classification**
- Thresholded 2019 price variance into `Good`, `Bad`, `Neutral`
- Achieved ~96% accuracy using Random Forest Classifier

---

## 🧠 Key Learnings

- Importance of removing **data leakage**
- Value of **feature importance** to interpret black-box models
- Difficulties in **stock price regression** vs. classification

---

## 📌 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn

---

## ✅ Final Thoughts

This project offers a strong foundation for:
- Financial modeling
- Feature engineering from economic data
- Building interpretable ML models

> Next step: Experiment with deep learning models or ensemble blending for better predictions.

---

## 📁 File Structure

- `US-Stocks.ipynb`: Main notebook with analysis
- `README.md`: This summary